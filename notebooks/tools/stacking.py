import numpy as np
import zarr

from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats, sigma_clip
from astropy.io import fits

from astropy.cosmology import Planck18
import astropy.units as u

from reproject import reproject_exact as reproject_exact # instead of exact for performance or interp

from photutils.aperture import (
    CircularAperture,
    CircularAnnulus,
    aperture_photometry,
    ApertureStats,
)
from photutils.centroids import centroid_com

import warnings

import pandas as pd

from scipy.interpolate import interp1d


PIXEL_SCALE_ARCSEC = 0.5
PIXEL_AREA_ARCSEC2 = PIXEL_SCALE_ARCSEC ** 2  # 0.25 arcsec^2

FLUX_UNIT = "1e-17 erg / s / cm^2 / AA  (pro Pixel, {:.2f} arcsec^2)".format(
    PIXEL_AREA_ARCSEC2
)
SB_UNIT = "1e-17 erg / s / cm^2 / AA / arcsec^2"


def flux_to_sb(cube_or_img):
    return np.asarray(cube_or_img, dtype=float) / PIXEL_AREA_ARCSEC2

from astropy.modeling.models import Moffat2D, Gaussian2D

def make_source_psf(shape, fwhm_arcsec, pixel_scale=0.5, beta=3.5):

    ny, nx = shape
    center_y, center_x = (ny - 1) / 2.0, (nx - 1) / 2.0
    fwhm_pix = fwhm_arcsec / pixel_scale
    
    y, x = np.mgrid[:ny, :nx]
    
    gamma = fwhm_pix / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))
    psf = Moffat2D(amplitude=1.0, x_0=center_x, y_0=center_y, gamma=gamma, alpha=beta)(x, y)
        
    return psf / np.nansum(psf)


def sb_to_flux(cube_or_img):
    return np.asarray(cube_or_img, dtype=float) * PIXEL_AREA_ARCSEC2


header = fits.Header.fromtextfile('/Users/bene/Desktop/mpe/fits/cubes/header.txt')
wcs = WCS(header)

COLNAMES = {
    "ra": ["ra", "RA", "Ra", "RAJ2000", "ra_vdfi", "ra_hetdex"],
    "dec": ["dec", "DEC", "Dec", "DEJ2000", "dec_vdfi", "dec_hetdex"],
    "z": ["z", "Z", "redshift", "REDSHIFT", "zspec", "ZSPEC", "z_vdfi", "z_hetdex", "redshift"],
    "flux": ["flux", "Flux", "FLUX", "flux_lya"],
    "luminosity": ["lum", "luminosity", "luminosity_lae", "LUMINOSITY"],
    "completeness": ["completeness", "comp", "COMPLETENESS"],
}


def _find_col(table, aliases):
    for name in aliases:
        if name in table.colnames:
            return name
    raise KeyError(f"Keine Spalte gefunden. Erwartet: {aliases} | Vorhanden: {table.colnames}")


def make_aperture_mask(shape, center, radius, ellipse_axes=None, theta=0.0):
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    yc, xc = center

    dx = xx - xc
    dy = yy - yc

    if ellipse_axes is not None:
        a, b = ellipse_axes
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        x_rot = dx * cos_t + dy * sin_t
        y_rot = -dx * sin_t + dy * cos_t
        mask = (x_rot / a) ** 2 + (y_rot / b) ** 2 <= 1.0
    else:
        mask = dx ** 2 + dy ** 2 <= radius ** 2

    return mask


def subtract_sky_sigmaclip(cube, mask=None, sigma=3.0, maxiters=5, stat="median"):

    data = np.asarray(cube, dtype=float)
    n_wave, ny, nx = data.shape

    if mask is None:
        excl = np.zeros((ny, nx), dtype=bool)
        excl_is_3d = False
    else:
        mask = np.asarray(mask, dtype=bool)
        excl_is_3d = mask.ndim == 3
        excl = mask

    sky_sub_cube = np.empty_like(data)

    for k in range(n_wave):
        sl = data[k]
        excl_k = excl[k] if excl_is_3d else excl

        valid = np.isfinite(sl) & (~excl_k)
        if not np.any(valid):
            sky_sub_cube[k] = sl
            continue

        mean_c, median_c, std_c = sigma_clipped_stats(sl[valid], sigma=sigma, maxiters=maxiters)
        sky_level = median_c if stat == "median" else mean_c
        sky_sub_cube[k] = sl - sky_level

    return sky_sub_cube


def prepare_subcube(ra, dec, z, zarr_cube, width=25, spec_width=25):
    spec = (z + 1) * 1216  # Å
    x, y, z_pix = wcs.all_world2pix(ra, dec, spec, 0)
    xi, yi, zi = int(round(float(x))), int(round(float(y))), int(round(float(z_pix)))

    n_wave, ny, nx = zarr_cube.shape

    z0, z1 = zi - spec_width, zi + spec_width
    y0, y1 = yi - width, yi + width
    x0, x1 = xi - width, xi + width

    out_of_bounds = (z0 < 0 or z1 > n_wave or y0 < 0 or y1 > ny or x0 < 0 or x1 > nx)

    if out_of_bounds:
        return None, None  # Quelle liegt nicht vollstaendig im Cube

    sub_cube = np.asarray(zarr_cube[z0:z1, y0:y1, x0:x1])

    sub_wcs = wcs.deepcopy()
    sub_wcs.wcs.crpix[0] -= x0
    sub_wcs.wcs.crpix[1] -= y0
    sub_wcs.wcs.crpix[2] -= z0

    return sub_cube, sub_wcs.celestial


def subtract_sky_per_slice(subcube):

    ny, nx = subcube.shape[1], subcube.shape[2]

    mask = make_aperture_mask(
        shape=(ny, nx),
        center=(ny // 2, nx // 2),
        radius=ny // 4,
    )

    sky_subtracted = subtract_sky_sigmaclip(
        subcube,
        mask=mask,
        sigma=3.0,
        maxiters=5,
        stat="median",
    )

    return sky_subtracted


def subtract_continuum(cube, line_mask=None, degree=4, sigma=3.0, maxiters=5):
    n_wave, ny, nx = cube.shape
    wave = np.arange(n_wave)
    if line_mask is None:
        line_mask = np.zeros(n_wave, dtype=bool)
        center = n_wave // 2
        line_mask[center - 9:center + 10] = True

    continuum = np.full_like(cube, np.nan)

    for j in range(ny):
        for i in range(nx):
            spec = cube[:, j, i]
            valid = np.isfinite(spec) & (~line_mask)
            if np.sum(valid) < degree + 1:
                continue

            x_fit, y_fit = wave[valid], spec[valid]

            # iteratives Sigma-Clipping auf die Residuen
            for _ in range(maxiters):
                coeffs = np.polyfit(x_fit, y_fit, deg=degree)
                resid = y_fit - np.polyval(coeffs, x_fit)
                clipped = sigma_clip(resid, sigma=sigma, maxiters=1)
                keep = ~clipped.mask
                if keep.sum() == len(x_fit):
                    break
                x_fit, y_fit = x_fit[keep], y_fit[keep]

            continuum[:, j, i] = np.polyval(coeffs, wave)

    cont_sub_cube = cube - continuum
    return cont_sub_cube


def make_narrowband(subcube, line_mask=None):
    
    n_wave, _, _ = subcube.shape
    wave = np.arange(n_wave)
    if line_mask is None:
        line_mask = np.zeros(n_wave, dtype=bool)
        center = n_wave // 2
        line_mask[center - 9:center + 10] = True

    selected = subcube[line_mask]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        image = np.nanmean(selected, axis=0)

    return image


def make_wcs(ra, dec, z, kpc_per_pixel=3, npix=50):
    kpc_per_arcsec = Planck18.kpc_proper_per_arcmin(z).to(u.kpc / u.arcsec).value
    arcsec_per_pixel = kpc_per_pixel / kpc_per_arcsec
    deg_per_pixel = arcsec_per_pixel / 3600.0

    w = WCS(naxis=2)
    w.wcs.crpix = [npix / 2, npix / 2]
    w.wcs.cdelt = [-deg_per_pixel, deg_per_pixel]
    w.wcs.crval = [ra, dec]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w


def scale_slice(img2d, wcs2d, target_wcs, npix):
    
    regrid, footprint = reproject_exact(
        (img2d, wcs2d), target_wcs, shape_out=(npix, npix)
    )
    return regrid, footprint

import numpy as np
import warnings

class StackingFromSubcubes:
    
    def __init__(self, subcube_list=None, kpc_pxl=2, npix=150, saved_cube=None, saved_vgrid=None, z_ref=None):
        if subcube_list and saved_cube is None:
            raise ValueError("subcube_list oder saved_cube darf nicht leer sein.")

        self.subcubes = subcube_list
        self.kpc_pxl = kpc_pxl
        self.npix = npix

        self.stacked_cube = saved_cube
        
        if self.subcubes is not None:
            self.n_wave = self.subcubes[0]["subcube"].shape[0]
        else:
            self.n_wave = self.stacked_cube.shape[0]

        self.rest_wave = 1215.67
        self.dv = 100

        self.v_grid = saved_vgrid
        self.z_ref = z_ref
        
        self.spec_width = self.n_wave // 2
        self.wave_pix = np.arange(self.n_wave) - self.spec_width


        self.stacked_psf = None

    def stack(self, do_sky_sub=False, do_cont_sub=False, normalize=False, verbose=True):
        z_ref = np.median([item["z"] for item in self.subcubes])

        cube_stack = []  # (n_sources, n_wave, npix, npix)
        foot_stack = []  # (n_sources, n_wave, npix, npix)
        n_skipped = 0

        for idx, item in enumerate(self.subcubes):
            subcube = np.copy(item["subcube"])
            sub_wcs = item["wcs"]
            ra, dec, z = item["ra"], item["dec"], item["z"]
            lum = item.get("lum", np.nan)

            if "mask" in item and item["mask"] is not None and np.any(item["mask"]):
                subcube[:, item["mask"]] = np.nan

            if normalize:
                if not np.isfinite(lum) or lum <= 0:
                    n_skipped += 1
                    continue

            if do_sky_sub:
                subcube = subtract_sky_per_slice(subcube)

            sb_cube = flux_to_sb(subcube) * ((1.0 + z_ref) / (1.0 + z)) ** 3

            if normalize:
                sb_cube = sb_cube / lum

            target_wcs = make_wcs(ra, dec, z, kpc_per_pixel=self.kpc_pxl, npix=self.npix)

            regridded = np.full((self.n_wave, self.npix, self.npix), np.nan)
            footprint = np.zeros((self.n_wave, self.npix, self.npix))

            for k in range(self.n_wave):
                regrid_k, foot_k = scale_slice(sb_cube[k], sub_wcs, target_wcs, self.npix)
                regridded[k] = regrid_k
                footprint[k] = foot_k

            cube_stack.append(regridded)
            foot_stack.append(footprint)

        if len(cube_stack) == 0:
            raise RuntimeError("Keine gültigen Cubes zum Stacken übrig.")

        cube_stack = np.array(cube_stack)
        foot_stack = np.array(foot_stack)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            valid = np.isfinite(cube_stack)
            foot_stack_masked = np.where(valid, foot_stack, 0.0)
            cube_filled = np.where(valid, cube_stack, 0.0)

            foot_sum = np.nansum(foot_stack_masked, axis=0)
            weighted_stack = np.where(
                foot_sum > 0,
                np.nansum(cube_filled * foot_stack_masked, axis=0) / foot_sum,
                np.nan
            )

        if verbose:
            print(f"Skipped: {n_skipped}")
            print(f"Stacked cube shape: {weighted_stack.shape}  (n_wave, npix, npix)")
            print(f"Stacked cube units: {SB_UNIT}")

        if do_cont_sub:
            weighted_stack = subtract_continuum(weighted_stack)

        self.stacked_cube = weighted_stack
        
        return weighted_stack

    def narrowband_from_cube(self, half_width=15, mode="sum", stacked_cube=None):
        if stacked_cube is None:
            if self.stacked_cube is None:
                raise RuntimeError("Noch kein gestacktes Cube vorhanden - erst stack() aufrufen.")
            stacked_cube = self.stacked_cube

        n_wave = stacked_cube.shape[0]
        center = n_wave // 2
        sel = slice(max(center - half_width, 0), min(center + half_width + 1, n_wave))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if mode == "mean":
                nb = np.nanmean(stacked_cube[sel], axis=0)
            elif mode == "sum":
                nb = np.nansum(stacked_cube[sel], axis=0) * 2
            elif mode == "median":
                nb = np.nanmedian(stacked_cube[sel], axis=0)
            else:
                raise ValueError("mode muss 'mean', 'sum' oder 'median' sein")

        return nb

    def stack(
        self,
        do_sky_sub=False,
        do_cont_sub=False,
        normalize=False,
        v_max=2000.0,        
        dv=100.0,           
        dlam_obs=2.0,        
        rest_wave=1215.67,   
        verbose=True,
    ):
        c_kms = 299792.458
        z_ref = np.median([item["z"] for item in self.subcubes])
        self.z_ref = z_ref
        self.rest_wave = rest_wave

        v_grid = np.arange(-v_max, v_max + dv * 0.5, dv)
        self.v_grid = v_grid
        self.dv = dv
        n_v = len(v_grid)

        cube_stack = []  # (n_sources, n_v, npix, npix)
        foot_stack = []  # (n_sources, n_v, npix, npix)
        n_skipped = 0

        for idx, item in enumerate(self.subcubes):
            subcube = np.copy(item["subcube"])
            sub_wcs = item["wcs"]
            ra, dec, z = item["ra"], item["dec"], item["z"]
            lum = item.get("lum", np.nan)

            if "mask" in item and item["mask"] is not None and np.any(item["mask"]):
                subcube[:, item["mask"]] = np.nan

            if normalize:
                if not np.isfinite(lum) or lum <= 0:
                    n_skipped += 1
                    continue

            if do_sky_sub:
                subcube = subtract_sky_per_slice(subcube)

            n_wave_sub = subcube.shape[0]
            center_pix = n_wave_sub // 2
            lam_obs_0 = (1.0 + z) * rest_wave
            dlam_arr = (np.arange(n_wave_sub) - center_pix) * dlam_obs
            v_orig = c_kms * (dlam_arr / lam_obs_0)

            sb_cube = flux_to_sb(subcube) * ((1.0 + z) / (1.0 + z_ref)) ** 5

            if normalize:
                sb_cube = sb_cube / lum

            interp_func = interp1d(
                v_orig,
                sb_cube,
                axis=0,
                kind="linear",
                bounds_error=False,
                fill_value=np.nan,
            )
            sb_cube_v = interp_func(v_grid)  # Shape: (n_v, ny_sub, nx_sub)

            target_wcs = make_wcs(ra, dec, z, kpc_per_pixel=self.kpc_pxl, npix=self.npix)

            regridded = np.full((n_v, self.npix, self.npix), np.nan)
            footprint = np.zeros((n_v, self.npix, self.npix))

            for k in range(n_v):
                regrid_k, foot_k = scale_slice(sb_cube_v[k], sub_wcs, target_wcs, self.npix)
                regridded[k] = regrid_k
                footprint[k] = foot_k

            cube_stack.append(regridded)
            foot_stack.append(footprint)

        if len(cube_stack) == 0:
            raise RuntimeError("Keine gültigen Cubes zum Stacken übrig.")

        cube_stack = np.array(cube_stack)
        foot_stack = np.array(foot_stack)

        # Footprint-gewichtete Mittelung
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            valid = np.isfinite(cube_stack)
            foot_stack_masked = np.where(valid, foot_stack, 0.0)
            cube_filled = np.where(valid, cube_stack, 0.0)

            foot_sum = np.nansum(foot_stack_masked, axis=0)
            weighted_stack = np.where(
                foot_sum > 0,
                np.nansum(cube_filled * foot_stack_masked, axis=0) / foot_sum,
                np.nan,
            )

        if verbose:
            print(f"Skipped: {n_skipped}")
            print(f"z_ref = {z_ref:.3f} | v_grid: [{v_grid[0]:.0f}, {v_grid[-1]:.0f}] km/s (dv={dv} km/s)")
            print(f"Stacked cube shape: {weighted_stack.shape}  (n_v, npix, npix)")
            print(f"Stacked cube units: {SB_UNIT} (skaliert auf z_ref)")

        if do_cont_sub:
            line_mask = np.abs(v_grid) <= 750.0
            weighted_stack = subtract_continuum(
                weighted_stack,
                line_mask=line_mask,
                degree=2,  
            )

        self.stacked_cube = weighted_stack
        return weighted_stack

    def narrowband_from_cube(self, v_half_width=500.0, mode="sum", stacked_cube=None):

        if stacked_cube is None:
            if self.stacked_cube is None:
                raise RuntimeError("Noch kein gestacktes Cube vorhanden - erst stack() aufrufen.")
            stacked_cube = self.stacked_cube

        c_kms = 299792.458
        sel = np.abs(self.v_grid) <= v_half_width

        dlam_ref = (1.0 + self.z_ref) * self.rest_wave * (self.dv / c_kms)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if mode == "mean":
                nb = np.nanmean(stacked_cube[sel], axis=0)
            elif mode == "sum":
                nb = np.nansum(stacked_cube[sel], axis=0) * dlam_ref
            elif mode == "median":
                nb = np.nanmedian(stacked_cube[sel], axis=0)
            else:
                raise ValueError("mode muss 'mean', 'sum' oder 'median' sein")

        return nb

    def narrowband_half(self, half_width=5, mode="sum", stacked_cube=None):
            if stacked_cube is None:
                if self.stacked_cube is None:
                    raise RuntimeError("Noch kein gestacktes Cube vorhanden - erst stack() aufrufen.")
                stacked_cube = self.stacked_cube
    
            n_wave = stacked_cube.shape[0]
            center = n_wave // 2
            sel = slice(center, min(center + half_width, n_wave))
            sel_blau = slice(max(center - half_width, 0), center)

    
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                if mode == "mean":
                    nb = np.nanmean(stacked_cube[sel], axis=0)
                    nb_blau = np.nanmean(stacked_cube[sel_blau], axis=0)
                elif mode == "sum":
                    nb = np.nansum(stacked_cube[sel], axis=0) * 2
                    nb_blau = np.nansum(stacked_cube[sel_blau], axis=0) * 2
                elif mode == "median":
                    nb = np.nanmedian(stacked_cube[sel], axis=0)
                    nb_blau = np.nanmedian(stacked_cube[sel_blau], axis=0)
                else:
                    raise ValueError("mode muss 'mean', 'sum' oder 'median' sein")
    
            return nb, nb_blau

    def narrowband_half(self, v_half_width=500.0, mode="sum", stacked_cube=None):

        if stacked_cube is None:
            if self.stacked_cube is None:
                raise RuntimeError(" kein gestackter Cube erst stack() aufrufen.")
            stacked_cube = self.stacked_cube

        if self.v_grid is None:
            raise RuntimeError(
                "Kein v_grid vorhanden - bitte v_grid in stack() setzen oder beim Laden übergeben."
            )

        v_grid = np.asarray(self.v_grid, dtype=float)
        if len(v_grid) != stacked_cube.shape[0]:
            raise ValueError(
                f"Länge von v_grid ({len(v_grid)}) passt nicht zur Cube-Achse ({stacked_cube.shape[0]})."
            )

        mask_red = (v_grid >= 0.0) & (v_grid <= v_half_width)
        mask_blue = (v_grid >= -v_half_width) & (v_grid < 0.0)

        if not np.any(mask_red) or not np.any(mask_blue):
            raise ValueError(
                f"v_half_width={v_half_width} km/s liegt außerhalb des verfügbaren v_grid "
                f"[{v_grid.min():.1f}, {v_grid.max():.1f}] km/s."
            )

        c_kms = 299792.458
        dv = self.dv if self.dv is not None else float(np.median(np.abs(np.diff(v_grid))))
        z_eff = self.z_ref if self.z_ref is not None else 0.0
        dlambda = (dv / c_kms) * self.rest_wave * (1.0 + z_eff)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if mode == "mean":
                nb = np.nanmean(stacked_cube[mask_red], axis=0)
                nb_blau = np.nanmean(stacked_cube[mask_blue], axis=0)
            elif mode == "sum":
                nb = np.nansum(stacked_cube[mask_red], axis=0) * dlambda * 2.0
                nb_blau = np.nansum(stacked_cube[mask_blue], axis=0) * dlambda * 2.0
            elif mode == "median":
                nb = np.nanmedian(stacked_cube[mask_red], axis=0)
                nb_blau = np.nanmedian(stacked_cube[mask_blue], axis=0)
            else:
                raise ValueError("mode muss 'mean', 'sum' oder 'median' sein")

        return nb, nb_blau
    
    def extract_sb_profile(
        self, img, error_map=None, center=None, r_min=1, r_max=40, n_bins=10, kpc_per_px=1.
    ):
        """Extrahiert ein 1D-Oberflaechenhelligkeitsprofil aus einem LAE-Stack."""
        if center is None:
            ny, nx = img.shape
            center = ((nx - 1) / 2.0, (ny - 1) / 2.0)

        r_edges = np.geomspace(r_min, r_max, n_bins+1)

        r_eff = []
        r_err = []  
        sb_profile = []
        sb_err = []

        for r_in, r_out in zip(r_edges[:-1], r_edges[1:]):
            annulus = CircularAnnulus(center, r_in=r_in, r_out=r_out)
            stats = ApertureStats(img, annulus, error=error_map)

            r_mid = np.sqrt(0.5 * (r_in**2 + r_out**2))
            r_eff.append(r_mid)
            
            r_err.append([r_mid - r_in, r_out - r_mid])

            sb_profile.append(stats.median)

            if error_map is not None:
                sb_err.append(stats.mean_error)
            else:
                sb_err.append(stats.std / np.sqrt(stats.sum_aper_area.value))

        r_scaled = np.array(r_eff) * kpc_per_px
        
        r_err_scaled = np.array(r_err).T * kpc_per_px

        return (
            r_scaled,
            r_err_scaled, 
            np.array(sb_profile),
            np.array(sb_err),
        )

    def stack_psf(self, fwhm=None):
            
        psf_stack = []
        foot_stack = []
        n_skipped = 0
            
        
            
        for idx, item in enumerate(self.subcubes):
            subcube = np.copy(item["subcube"])
            ra, dec, z = item["ra"], item["dec"], item["z"]   
            sub_wcs = item["wcs"]

            ny_sub, nx_sub = subcube.shape[0], subcube.shape[0]

            if fwhm is not None:
                fwhm = fwhm
            else:
                fwhm = 2.

            psf_val = item.get("psf", None)

            if psf_val is not None and np.isfinite(psf_val) and psf_val > 0:
                fwhm = float(psf_val)
                
            psf_raw = make_source_psf(
                shape=(ny_sub, nx_sub),
                fwhm_arcsec=fwhm,
            )
                
            target_wcs = make_wcs(ra, dec, z, kpc_per_pixel=self.kpc_pxl, npix=self.npix)
            regrid_psf, foot_psf = scale_slice(psf_raw, sub_wcs, target_wcs, self.npix)
            s = np.nansum(regrid_psf)
            if s>0:
                regrid_psf /= s
                    
            psf_stack.append(regrid_psf)
            foot_stack.append(foot_psf)
                
        psf_stack = np.array(psf_stack)
        foot_stack = np.array(foot_stack)
                
        weighted_psf = np.nansum(psf_stack * foot_stack, axis=0) / np.nansum(foot_stack, axis=0)
                
        self.stacked_psf = weighted_psf / np.nansum(weighted_psf)
                
        return self.stacked_psf

    def smooth_cont(self, line_idx=None, half_mask=10, window=45, min_periods=5, axis=0):
        cube = np.moveaxis(np.asarray(self.stacked_cube, dtype=float), axis, 0)
        n_lam, n1, n2 = cube.shape

        if line_idx is None:
            line_idx = n_lam // 2

        masked = cube.copy()
        start = max(0, line_idx - half_mask)
        stop = min(n_lam, line_idx + half_mask + 1)
        masked[start:stop, :, :] = np.nan

        df = pd.DataFrame(masked.reshape(n_lam, -1))
        continuum_2d = (
            df.rolling(window=window, center=True, min_periods=min_periods)
            .median()
            .to_numpy()
        )

        continuum = continuum_2d.reshape(n_lam, n1, n2)
        cube_sub = cube - continuum

        cube_sub = np.moveaxis(cube_sub, 0, axis)
        continuum = np.moveaxis(continuum, 0, axis)

        self.stacked_cube_sub = cube_sub
        return cube_sub, continuum
    
def load_subcubes_npz(filename="subcubes_for_laptop.npz"):
    data = np.load(filename, allow_pickle=True)
    n_items = int(data["n_items"])
    subcube_list = []

    for i in range(n_items):
        hdr = fits.Header.fromstring(str(data[f"wcs_hdr_{i}"]))
        item = {
            "subcube": data[f"cube_{i}"],
            "wcs": WCS(hdr),
            "ra": float(data[f"ra_{i}"]),
            "dec": float(data[f"dec_{i}"]),
            "z": float(data[f"z_{i}"]),
            "lum": float(data[f"lum_{i}"]),
            "mask": data[f"mask_{i}"] if f"mask_{i}" in data else None,
        }
        subcube_list.append(item)
        
    print(f"{len(subcube_list)} Subcubes erfolgreich geladen.")
    return subcube_list