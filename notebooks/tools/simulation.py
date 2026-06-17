import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from math import ceil
from regions import Regions

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
Z_REF = 2.7

LYA_REST = 1215.67  # consistent with plotting code's rest_wavelength

# Maximaler Stempel-Radius (muss zu MAX_PAD in generate_source_stamp passen)
MAX_PAD = 15
# Mindestabstand zum Rand, damit der Stempel nie geclippt wird
EDGE_MARGIN = MAX_PAD + 1


def slice_from_z(z):
    crpix = 1
    crval = 3470.
    cdelt = 2.
    obs_wave = LYA_REST * (1. + z)
    slice_idx = crpix + (obs_wave - crval) / cdelt
    return slice_idx


def z_from_slice(slice_idx):
    crpix = 1
    crval = 3470.
    cdelt = 2.
    obs_wave = crval + (slice_idx - crpix) * cdelt
    z = (obs_wave / LYA_REST) - 1.0
    return z


def scale_params(z, flux, sigma):
    if z <= 0 or not np.isfinite(z):
        return np.nan, np.nan

    try:
        DL = cosmo.luminosity_distance(Z_REF).value
        DA = cosmo.angular_diameter_distance(Z_REF).value
        dl = cosmo.luminosity_distance(z).value
        da = cosmo.angular_diameter_distance(z).value
    except Exception:
        return np.nan, np.nan

    if dl <= 0 or da <= 0 or not np.isfinite(dl) or not np.isfinite(da):
        return np.nan, np.nan

    flux_scaled  = flux  * (DL / dl) ** 2
    sigma_scaled = sigma * (DA / da)

    if not np.isfinite(flux_scaled) or not np.isfinite(sigma_scaled) or sigma_scaled <= 0:
        return np.nan, np.nan

    return flux_scaled, sigma_scaled


def generate_source_stamp(z, theta, flux, sigma, sigma_lam, elipticity):
    flux_s, sigma_s = scale_params(z, flux, sigma)

    if (np.isnan(flux_s) or np.isnan(sigma_s) or sigma_s <= 0 or not np.isfinite(sigma_s)):
        return None, 0, 0

    if not np.isfinite(elipticity) or elipticity <= 0.05:
        elipticity = 1

    # Sigma begrenzen, damit der Stempel nicht groesser als MAX_PAD wird
    if int(ceil(sigma_s * 5)) > MAX_PAD:
        sigma_s = MAX_PAD / 5.0

    pad_spat = int(ceil(sigma_s * 5))
    cdelt    = 2.0
    sig_lam_pix = (sigma_lam / cdelt) / 2.355
    if not np.isfinite(sig_lam_pix) or sig_lam_pix <= 0:
        return None, 0, 0
    pad_spec    = int(ceil(sig_lam_pix * 5))

    if pad_spec > MAX_PAD:
        pad_spec = MAX_PAD
        sig_lam_pix = MAX_PAD / 5.0

    z_axis = np.arange(-pad_spec, pad_spec + 1)
    y_axis = np.arange(-pad_spat, pad_spat + 1)
    x_axis = np.arange(-pad_spat, pad_spat + 1)
    z_grid, y_grid, x_grid = np.ix_(z_axis, y_axis, x_axis)

    theta_rad = np.radians(theta)
    x_rot =  x_grid * np.cos(theta_rad) + y_grid * np.sin(theta_rad)
    y_rot = -x_grid * np.sin(theta_rad) + y_grid * np.cos(theta_rad)

    a = sigma_s
    b = sigma_s * elipticity

    if a <= 0 or b <= 0 or not np.isfinite(a) or not np.isfinite(b):
        return None, 0, 0

    spatial  = np.exp(-((x_rot**2 / (2*a**2)) + (y_rot**2 / (2*b**2))))
    spectral = np.exp(-(z_grid**2 / (2*sig_lam_pix**2)))
    stamp    = spatial * spectral

    total_sum = np.sum(stamp)
    if total_sum <= 0 or not np.isfinite(total_sum):
        return None, 0, 0

    stamp = (stamp / total_sum) * flux_s

    if not np.all(np.isfinite(stamp)):
        return None, 0, 0

    return stamp.astype(np.float32), pad_spec, pad_spat


def build_fov_mask(nz, ny, nx, wcs, reg_file=None):
    mask = np.ones((ny, nx), dtype=bool)

    if reg_file is not None:
        regs = Regions.read(reg_file, format='ds9')

        include_regions = [r for r in regs if r.meta.get('include', True)]
        exclude_regions = [r for r in regs if not r.meta.get('include', True)]

        if include_regions:
            inc_mask = np.zeros((ny, nx), dtype=bool)
            for reg in include_regions:
                pix_reg = reg.to_pixel(wcs.celestial) if hasattr(reg, 'to_pixel') else reg
                inc_mask |= pix_reg.to_mask(mode='center').to_image((ny, nx)).astype(bool)
            mask &= inc_mask

        if exclude_regions:
            for reg in exclude_regions:
                pix_reg = reg.to_pixel(wcs.celestial) if hasattr(reg, 'to_pixel') else reg
                exc = pix_reg.to_mask(mode='center').to_image((ny, nx)).astype(bool)
                mask &= ~exc

    return mask


def pixel_in_fov(x_pix, y_pix, fov_mask):
    xi, yi = int(round(x_pix)), int(round(y_pix))
    ny, nx = fov_mask.shape
    if xi < 0 or xi >= nx or yi < 0 or yi >= ny:
        return False
    return bool(fov_mask[yi, xi])


def generate_positions(
    nz, ny, nx, wcs,
    existing_catalog,
    chunks,
    n_per_bin=10,
    n_bins=10,
    matching_radius_arcsec=10.0,
    fov_mask=None,
):
    ra_ax = dec_ax = wave_ax = None
    for i, ctype in enumerate(wcs.wcs.ctype):
        cu = ctype.upper()
        if 'RA'   in cu: ra_ax   = i
        elif 'DEC' in cu: dec_ax  = i
        elif any(k in cu for k in ('WAVE', 'AWAV', 'FREQ')): wave_ax = i
    if None in (ra_ax, dec_ax, wave_ax):
        raise ValueError(f"WCS-Achsen nicht eindeutig: {list(wcs.wcs.ctype)}")

    def pix_to_world(x_pix, y_pix, z_pix):
        pix = [0.0, 0.0, 0.0]
        pix[ra_ax]   = float(x_pix)
        pix[dec_ax]  = float(y_pix)
        pix[wave_ax] = float(z_pix)
        world = wcs.all_pix2world([pix], 0)[0]
        ra_val, dec_val = world[ra_ax], world[dec_ax]
        if not (np.isfinite(ra_val) and np.isfinite(dec_val)):
            return np.nan, np.nan
        return ra_val, dec_val

    if len(existing_catalog) > 0:
        col_ra  = np.array([src[0] for src in existing_catalog], dtype=float)
        col_dec = np.array([src[1] for src in existing_catalog], dtype=float)
        col_z   = np.array([src[2] for src in existing_catalog], dtype=float)
    else:
        col_ra  = np.empty(0, dtype=float)
        col_dec = np.empty(0, dtype=float)
        col_z   = np.empty(0, dtype=float)

    existing_coords = None
    if len(col_ra) > 0:
        existing_coords = SkyCoord(ra=col_ra*u.deg, dec=col_dec*u.deg)

    positions = []
    for chunk_idx, (chunk_start, chunk_end) in enumerate(chunks):
        chunk_start = max(chunk_start, 0)
        chunk_end   = min(chunk_end, nz)
        z_lo = max(chunk_start, EDGE_MARGIN)
        z_hi = min(chunk_end, nz - EDGE_MARGIN)
        if z_hi <= z_lo:
            print(f"Chunk {chunk_idx} ({chunk_start}-{chunk_end}) liegt komplett im Randbereich, "
                  f"wird uebersprungen.")
            continue

        bin_edges = np.linspace(z_lo, z_hi, n_bins + 1)

        for bin_idx in range(n_bins):
            bin_z_lo = bin_edges[bin_idx]
            bin_z_hi = bin_edges[bin_idx + 1]

            generated = 0
            attempts  = 0
            max_attempts = n_per_bin * 50

            while generated < n_per_bin and attempts < max_attempts:
                attempts += 1
                z_pix = np.random.uniform(bin_z_lo, bin_z_hi)
                z = z_from_slice(z_pix)
                if z <= 0 or not np.isfinite(z):
                    continue

                x_pix = np.random.uniform(EDGE_MARGIN, nx - EDGE_MARGIN)
                y_pix = np.random.uniform(EDGE_MARGIN, ny - EDGE_MARGIN)

                if fov_mask is not None and not pixel_in_fov(x_pix, y_pix, fov_mask):
                    continue

                ra_val, dec_val = pix_to_world(x_pix, y_pix, z_pix)
                if not (np.isfinite(ra_val) and np.isfinite(dec_val)):
                    continue

                if existing_coords is not None:
                    new_coord = SkyCoord(ra=ra_val*u.deg, dec=dec_val*u.deg)
                    sep = new_coord.separation(existing_coords).arcsec
                    if np.any((sep < matching_radius_arcsec) & (np.abs(col_z - z) < 0.05)):
                        continue

                positions.append({
                    'chunk_index': chunk_idx,
                    'bin_index':   bin_idx,
                    'x_pix': x_pix,
                    'y_pix': y_pix,
                    'z_pix': z_pix,
                    'z':     z,
                    'ra':    ra_val,
                    'dec':   dec_val,
                })
                generated += 1

            if generated < n_per_bin:
                print(f"Warnung: Chunk {chunk_idx}, Bin {bin_idx} ({bin_z_lo:.1f}-{bin_z_hi:.1f}): "
                      f"nur {generated}/{n_per_bin} Positionen gefunden "
                      f"(nach {attempts} Versuchen).")

    return positions



def inject_sources_into_cube(
    existing_cube,
    positions,
    flux_bins,
    sigma_range=(1, 10),
    sigma_lam=10,
    elipticity_range=(0.7, 1),
    theta_range=(0, 180),
):
    """
    Injiziert fuer jede Position in `positions` eine Quelle mit Fluss aus
    `flux_bin = (bin_min, bin_max)`. Rotation, sigma und elipticity werden
    zufaellig aus den uebergebenen Bereichen gezogen.

    Stempel, die nicht vollstaendig ins Cube passen, werden uebersprungen
    (kein Clipping), um keine abgeschnittenen ("Eckenform") Quellen zu erzeugen.

    Gibt das modifizierte Cube und einen Katalog der injizierten Quellen zurueck.
    """
    nz, ny, nx = existing_cube.shape
    new_sources_catalog = []
    source_counter = 0

    for pos in positions:
        flux = np.random.uniform(*flux_bins[pos["bin_index"]])
        theta      = np.random.uniform(*theta_range)
        sigma      = np.random.uniform(*sigma_range)
        elipticity = np.random.uniform(*elipticity_range)

        stamp_res = generate_source_stamp(
            pos['z'], theta, flux, sigma, sigma_lam, elipticity
        )
        if stamp_res[0] is None:
            continue

        stamp, p_spec, p_spat = stamp_res

        z_int = int(round(pos['z_pix']))
        y_int = int(round(pos['y_pix']))
        x_int = int(round(pos['x_pix']))

        z_min, z_max = z_int - p_spec, z_int + p_spec + 1
        y_min, y_max = y_int - p_spat, y_int + p_spat + 1
        x_min, x_max = x_int - p_spat, x_int + p_spat + 1

        # Stempel komplett im Cube? Falls nicht, ueberspringen statt clippen.
        if z_min < 0 or z_max > nz or y_min < 0 or y_max > ny or x_min < 0 or x_max > nx:
            continue

        existing_cube[z_min:z_max, y_min:y_max, x_min:x_max] += stamp

        new_sources_catalog.append({
            'id':          source_counter,
            'chunk_index': pos['chunk_index'],
            "bin_index" : pos["bin_index"],
            'ra':          pos['ra'],
            'dec':         pos['dec'],
            'z':           pos['z'],
            'flux':        flux,
        })
        source_counter += 1

    return existing_cube, new_sources_catalog


def save_cube_fits(cube, outpath, header_params=None):
    hdu = fits.PrimaryHDU(cube.astype(np.float32))
    if header_params:
        for key, value in header_params.items():
            hdu.header[key] = value
    hdu.writeto(outpath, overwrite=True)
    print(f"Cube saved to {outpath}")


fits_header = {
        "CTYPE1": "RA---TAN", "CRPIX1": 1154.0, "CRVAL1": 334.3046, "CDELT1": -0.00013, "CUNIT1": "deg",
        "CTYPE2": "DEC--TAN", "CRPIX2": 1244.0, "CRVAL2": 0.2790, "CDELT2": 0.00013, "CUNIT2": "deg",
        "CTYPE3": "WAVE", "CRPIX3": 1.0, "CRVAL3": 3470.0, "CDELT3": 2.0, "CUNIT3": "Angstrom"
    }


