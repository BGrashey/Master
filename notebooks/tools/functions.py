import math
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
import warnings


import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord, search_around_sky
from astropy import units as u
from regions import Regions

from astropy.cosmology import Planck18 as cosmo
from astropy.cosmology import z_at_value

from scipy.special import erf
from scipy.optimize import curve_fit

from scipy.spatial import cKDTree

#plt.style.use("~/Desktop/Notebooks/thesis.mplstyle")


COLNAMES = {
    "ra":  ["ra_vdfi", "ra", "RA", "Ra", "RAJ2000", "ra_vdfi", "ra_hetdex"],
    "dec": ["dec_vdfi", "dec", "DEC", "Dec", "DEJ2000", "dec_vdfi", "dec_hetdex"],
    "z":   ["z_vdfi", "z", "Z", "redshift", "REDSHIFT", "zspec", "ZSPEC", "z_spec", "z_vdfi", "z_hetdex", "redshift"],
    "flux":    ["flux", "Flux", "FLUX", "flux_lya"],
    "luminosity": ["lum", "luminosity", "luminosity_lae", "LUMINOSITY"],
    "completeness": ["completeness", "comp", "COMPLETENESS"],
    "completeness_err": ["COMPLETENESS_ERR", "COMP_ERR"],
}
 
def _find_col(table, aliases):
    for name in aliases:
        if name in table.colnames:
            return name
    raise KeyError(f"Keine Spalte gefunden. Erwartet: {aliases} | Vorhanden: {table.colnames}")


def plot_cutout_grid(
    catalog,
    cube_file,
    output_file,
    col_seg         = None,          
    # Filter
    filters         = None,  
    # Grid-Optionen
    cutout_size     = 20,
    cols_in_grid    = 5,
    num_cutouts     = 25,
    num_wave_slices = 0,
    rest_wavelength = 1215.67,
    # Stretch
    use_fixed_stretch = True,
    vmin_fixed      = -2.0,
    vmax_fixed      =  1.0,
    cmap            = "inferno",
    # Titel
    title           = None,
    label           = None,         
):
    """
    Erstellt ein Cutout-Grid aus einem Katalog und einem FITS-Cube.

    Parameter
    ---------
    catalog_file      : str   Pfad zur FITS-Katalog-Datei
    cube_file         : str   Pfad zur FITS-Cube-Datei
    output_file       : str   Pfad zur Ausgabe-PDF/PNG
    col_ra/dec/z      : str   Spaltenname für RA, Dec, Redshift
    col_seg           : str   Spaltenname für Segment-ID (optional)
    filters           : list  Liste von Filter-Dicts:
                              [{"column": "x", "op": "==", "value": 0}]
    cutout_size       : int   Ausschnitt in Pixel
    cols_in_grid      : int   Spalten im Grid
    rest_wavelength   : float Ruhe-Wellenlänge in Ångström
    use_fixed_stretch : bool  True = feste vmin/vmax, False = Percentile
    vmin_fixed/vmax_fixed : float  Stretch-Grenzen falls use_fixed_stretch
    cmap              : str   Colormap
    title             : str   Titel (default: Katalogname)
    """

    if title is None:
        title = "Catalog"


    def redshift_to_slice(redshift):
        lya_obs = 1216.0 * (1.0 + redshift)
        slice_ = int(round(1 + (lya_obs - 3470) / 2))
        return slice_

    def sky_to_pixel(ra, dec, wcs):
        wcs2d = wcs.celestial
        x, y = wcs2d.all_world2pix(ra, dec, 0)
        return float(x), float(y)

    def extract_cutout(cube_data, x, y, z_slice, size):
        nz, ny, nx = cube_data.shape
        half = size // 2
        if z_slice < 0 or z_slice >= nz:
            return None, z_slice

        z0 = max(z_slice - num_wave_slices, 0)
        z1 = min(z_slice + num_wave_slices + 1, nz)

        x0, x1 = int(round(x)) - half, int(round(x)) - half + size
        y0, y1 = int(round(y)) - half, int(round(y)) - half + size

        n_sl = z1 - z0
        cutout_3d = np.full((n_sl, size, size), np.nan, dtype=float)

        sx0, sx1 = max(x0, 0), min(x1, nx)
        sy0, sy1 = max(y0, 0), min(y1, ny)
        cx0, cx1 = sx0 - x0, (sx0 - x0) + (sx1 - sx0)
        cy0, cy1 = sy0 - y0, (sy0 - y0) + (sy1 - sy0)

        if sx1 > sx0 and sy1 > sy0:
            cutout_3d[:, cy0:cy1, cx0:cx1] = cube_data[z0:z1, sy0:sy1, sx0:sx1]

        center_vals = np.nanmax(cutout_3d[:, half-1:half+2, half-1:half+2], axis=(1, 2))

        if np.all(np.isnan(center_vals)):
            return None, z_slice

        best_idx = int(np.nanargmax(center_vals))
        best_z_slice = z0 + best_idx

        return cutout_3d[best_idx], best_z_slice

    def apply_filters(cat, filters):
        mask = np.ones(len(cat), dtype=bool)
        op_map = {
            "==":     lambda a, b: a == b,
            "!=":     lambda a, b: a != b,
            "<":      lambda a, b: a <  b,
            "<=":     lambda a, b: a <= b,
            ">":      lambda a, b: a >  b,
            ">=":     lambda a, b: a >= b,
            "in":     lambda a, b: np.isin(a, b),
            "not in": lambda a, b: ~np.isin(a, b),
        }
        for f in filters:
            col, op, val = f["column"], f["op"], f["value"]
            if op not in op_map:
                raise ValueError(f"Unbekannter Operator '{op}'.")
            cond = op_map[op](cat[col], val)
            mask &= cond
            print(f"  Filter {col} {op} {val!r:20s} → {cond.sum()} / {len(cat)}")
        return mask


    """print(f"\nLade Katalog: {catalog_file}")
    with fits.open(catalog_file) as hdul:
        cat = hdul[1].data
    print(f"  {len(cat)} Objekte geladen.")"""

    cat = catalog[:num_cutouts]
    
    col_ra = _find_col(cat, COLNAMES["ra"])
    col_dec = _find_col(cat, COLNAMES["dec"])
    col_z = _find_col(cat, COLNAMES["z"])

    if filters:
        print(f"  Wende {len(filters)} Filter an:")
        mask = apply_filters(cat, filters)
        cat  = cat[mask]
        print(f"  → {len(cat)} Objekte nach Filterung.")

    ra_arr  = cat[col_ra]
    dec_arr = cat[col_dec]
    z_arr   = cat[col_z]
    seg_arr = cat[col_seg] if col_seg else [None] * len(cat)


    print(f"Lade Cube: {cube_file}")
    with fits.open(cube_file, memmap=True) as hdul:
        cube_data = hdul[0].data
        wcs       = WCS(hdul[0].header)
    print(f"  Shape: {cube_data.shape}")


    n_obj = len(ra_arr)
    ncols = cols_in_grid
    nrows = math.ceil(n_obj / ncols)
    fig_height = 6.102 / 3 * nrows

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(6.102, fig_height),
        facecolor="white",
        layout="constrained",
    )
    axes_flat = np.array(axes).reshape(-1)

    for ax in axes_flat:
        ax.set_facecolor("white")
        ax.axis("off")

    print(f"  Erzeuge {n_obj} Cutouts ...")

    for i, (ra, dec, z, seg) in enumerate(zip(ra_arr, dec_arr, z_arr, seg_arr)):
        ax = axes_flat[i]
        x_pix = y_pix = z_slice = cutout = None

        try:
            x_pix, y_pix = sky_to_pixel(ra, dec, wcs)
            z_slice = redshift_to_slice(z)
            cutout, z_slice = extract_cutout(cube_data, x_pix, y_pix, z_slice, cutout_size)
        except Exception as e:
            print(f"    Objekt {i+1}: Fehler – {e}")

        ax.axis("on")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#555555")
            spine.set_linewidth(0.6)

        if cutout is None or np.all(np.isnan(cutout)):
            ax.set_facecolor("white")
            ax.text(0.5, 0.5, "N/A", color="gray", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9)
        else:
            vmin = vmin_fixed if use_fixed_stretch else np.nanpercentile(cutout, 1)
            vmax = vmax_fixed if use_fixed_stretch else np.nanpercentile(cutout, 99)
            last_im = ax.imshow(cutout, origin="lower", cmap=cmap,
                      vmin=vmin, vmax=vmax, interpolation="nearest")
            if label and cat[label][i]:
                circle = plt.Circle((cutout_size/2, cutout_size/2), radius=3,
                         color="lime", fill=False, linewidth=1.2)
                ax.add_patch(circle)

        seg_str  = f"{int(seg)}" if seg is not None else "–"
        obs_wave = rest_wavelength * (1.0 + z)
        ax.set_title(
            rf"$\lambda_\mathrm{{obs}}$={obs_wave:.2f} Å",
            color="black", fontsize=11,
        )

        coord_label = (f"x={int(round(x_pix))}  y={int(round(y_pix))}  sl={z_slice}"
                       if x_pix is not None else "x=?  y=?  sl=?")
        ax.text(0.5, -0.04, coord_label, color="black",
                ha="center", va="top", transform=ax.transAxes,
                fontsize=11)

    if last_im is not None:
        cbar = fig.colorbar(
            last_im,
            ax=axes_flat.tolist(),
            orientation="vertical",
            fraction=0.02,     
            pad=0.03,         
            shrink=0.8         
        )
        cbar.ax.tick_params(labelsize=11)
        cbar.set_label("S/N", fontsize=11)

    fig.suptitle(
        f"λ_rest={rest_wavelength:.2f} Å",
        color="white", fontsize=11, y=1.01,
    )
    plt.savefig(output_file, dpi=300, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig)
    print(f"  Gespeichert: {output_file}")

    
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u

def completeness_analysis(injected_path, detected_path, sky_radius=3.0, dz=0.5, 
                         inj_col_ra="ra", det_col_ra="ra",
                          inj_col_dec="dec", det_col_dec="dec",
                         inj_col_z="z", det_col_z="z"):
    inj = Table.read(injected_path)
    det = Table.read(detected_path)
    
    coords_inj = SkyCoord(ra=inj[inj_col_ra].data * u.deg, dec=inj[inj_col_dec].data * u.deg)
    coords_det = SkyCoord(ra=det[det_col_ra].data * u.deg, dec=det[det_col_dec].data * u.deg)
    
    idx, sep, _ = coords_inj.match_to_catalog_sky(coords_det)
    
    delta_z = np.abs(np.array(inj[inj_col_z]) - np.array(det[det_col_z])[idx])
    
    inj["detected"] = (sep.arcsec <= sky_radius) & (delta_z <= dz)
    
    inj.write(injected_path.replace(".fits", "_flagged.fits"), overwrite=True)
    return inj



 
def _load(path):
    fmt = "fits" if path.endswith((".fits", ".fit")) else "csv"
    return Table.read(path, format=fmt)
 
def crossmatch(path1, path2, rad=1.0, dz_max=0.007, use_z=True):
    t1, t2 = _load(path1), _load(path2)
    
    t2 = t2[t2["n_spax"] >= 6]

    cat1 = SkyCoord(
        ra=t1[_find_col(t1, COLNAMES["ra"])].data * u.deg,
        dec=t1[_find_col(t1, COLNAMES["dec"])].data * u.deg,
    )
    cat2 = SkyCoord(
        ra=t2[_find_col(t2, COLNAMES["ra"])].data * u.deg,
        dec=t2[_find_col(t2, COLNAMES["dec"])].data * u.deg,
    )

    idx_t1, idx_t2, _, _ = search_around_sky(cat1, cat2, rad * u.arcsec)

    if use_z:
        z1 = np.asarray(t1[_find_col(t1, COLNAMES["z"])])
        z2 = np.asarray(t2[_find_col(t2, COLNAMES["z"])])

        mask = np.abs(z1[idx_t1] - z2[idx_t2]) < dz_max
        matched_idx = np.unique(idx_t1[mask])
    else:
        matched_idx = np.unique(idx_t1)

    t1["MATCHED"] = np.zeros(len(t1), dtype=bool)
    t1["MATCHED"][matched_idx] = True

    return t1


from astropy.table import MaskedColumn

def crossmatch_add(path1, path2, rad=1.0, dz_max=0.007, use_z=True, t2_prefix="t2_"):
    t1, t2 = _load(path1), _load(path2)

    ra1_col = _find_col(t1, COLNAMES["ra"])
    dec1_col = _find_col(t1, COLNAMES["dec"])
    ra2_col = _find_col(t2, COLNAMES["ra"])
    dec2_col = _find_col(t2, COLNAMES["dec"])

    cat1 = SkyCoord(ra=t1[ra1_col].data * u.deg, dec=t1[dec1_col].data * u.deg)
    cat2 = SkyCoord(ra=t2[ra2_col].data * u.deg, dec=t2[dec2_col].data * u.deg)

    # cat1 -> idx_t1 | cat2 -> idx_t2
    idx_t1, idx_t2, d2d, _ = search_around_sky(cat1, cat2, rad * u.arcsec)

    if use_z:
        z1 = np.asarray(t1[_find_col(t1, COLNAMES["z"])])
        z2 = np.asarray(t2[_find_col(t2, COLNAMES["z"])])
        mask = np.abs(z1[idx_t1] - z2[idx_t2]) < dz_max
        idx_t1 = idx_t1[mask]
        idx_t2 = idx_t2[mask]
        d2d = d2d[mask]

    best = {}
    for i1, i2, sep in zip(idx_t1, idx_t2, d2d.arcsec):
        if i1 not in best or sep < best[i1][1]:
            best[i1] = (i2, sep)

    matched_idx = np.array(sorted(best.keys()), dtype=int)

    t1["MATCHED"] = np.zeros(len(t1), dtype=bool)
    t1["MATCHED"][matched_idx] = True

    t1["dRA_arcsec"] = np.full(len(t1), np.nan)
    t1["dDec_arcsec"] = np.full(len(t1), np.nan)

    for colname in t2.colnames:
        new_col = f"{t2_prefix}{colname}"
        col_data = t2[colname]
        t1[new_col] = MaskedColumn(
            data=np.zeros(len(t1), dtype=col_data.dtype),
            mask=np.ones(len(t1), dtype=bool),
        )

    for i1, (i2, _) in best.items():
        ra1, dec1 = t1[ra1_col][i1], t1[dec1_col][i1]
        ra2, dec2 = t2[ra2_col][i2], t2[dec2_col][i2]

        t1["dRA_arcsec"][i1] = (ra1 - ra2) * np.cos(np.deg2rad(dec1)) * 3600.0
        t1["dDec_arcsec"][i1] = (dec1 - dec2) * 3600.0

        for colname in t2.colnames:
            new_col = f"{t2_prefix}{colname}"
            t1[new_col][i1] = t2[colname][i2]
            t1[new_col].mask[i1] = False

    return t1


def match_survey_footprint(catalog: str,
                           region="/data/hetdex/u/bgrashey/regions/fov.reg",
                           header="/data/hetdex/u/bgrashey/cubes/test.fits",
                          ):
    
    catalog = Table.read(catalog)
    
    with fits.open(header) as f:
        wcs = WCS(f[0].header).celestial
        
        
    ra = _find_col(catalog, COLNAMES["ra"])
    dec = _find_col(catalog, COLNAMES["dec"])
    z = _find_col(catalog, COLNAMES["z"])
    
    coords = SkyCoord(ra=catalog[ra].data * u.deg,
                      dec=catalog[dec].data * u.deg,
                      frame="fk5"
                     )
    
    regions = Regions.read(region, format="ds9")
    
    ii = np.zeros(len(coords), dtype=bool)
    ee = np.zeros(len(coords), dtype=bool)
    
    for r in regions:
        if r.meta.get("include") == 1:
            ii |= r.contains(coords, wcs)
        else:
            ee |= ~r.contains(coords, wcs)
 
    min_z = 4584 / 1215.67 - 1
    max_z = 5382 / 1215.67 - 1
    
    mask_z = (catalog[z] > min_z) & (catalog[z] < max_z)  
    
    mask = mask_z * ~ee * ii
    
    catalog = catalog[mask]
    
    return catalog
    
    


def z_max_fluxlim(flux_lim, flux, redshift, survey_min=2.7, survey_max=3.4):
    if flux < flux_lim:
        z = redshift
    else:
        dl = np.sqrt(flux / flux_lim) * cosmo.luminosity_distance(redshift)
        z = z_at_value(cosmo.luminosity_distance, dl)
    
    return np.clip(z, survey_min, survey_max)



def v_max(flux, flux_lim, redshift, area=1000*u.deg**2, survey_min=2.7, survey_max=3.4):
    z_min = survey_min
    z_max = z_max_fluxlim(flux_lim, flux, redshift, survey_min, survey_max)
    area_sr = area.to(u.sr).value
    
    v_max_ = cosmo.comoving_volume(z_max)
    v_min = cosmo.comoving_volume(z_min)

    volume = (v_max_ - v_min) * (area_sr / (4 * np.pi))

    return volume




def luminosity_function(flux_lim, catalog, bins,
                        area=0.07056104808102222*u.deg**2,
                        survey_min=2.77,
                        survey_max=3.43,
                        min_n=3):
    
    cat = Table.read(catalog)

    phi = []
    phi_err = []
    bin_centers = []
    n_gal = []

    col_z = _find_col(cat, COLNAMES["z"])
    col_flux = _find_col(cat, COLNAMES["flux"])
    col_lum = _find_col(cat, COLNAMES["luminosity"])
    col_comp = _find_col(cat, COLNAMES["completeness"])
    col_comp_err = _find_col(cat, COLNAMES["completeness_err"])

    bin_edges = bins
    num_bins = len(bin_edges) - 1

    for i in range(num_bins):
        lum_low = bin_edges[i]
        lum_high = bin_edges[i+1]
        
        delta_log_lum = np.log10(lum_high) - np.log10(lum_low)
        
        if i == num_bins - 1:
            mask = (cat[col_lum] >= lum_low) & (cat[col_lum] <= lum_high)
        else:
            mask = (cat[col_lum] >= lum_low) & (cat[col_lum] < lum_high)
            
        cat_ = cat[mask]
        
        if len(cat_) > 0:
            bin_center = np.median(cat_[col_lum]) 
        else:
            bin_center = 10**((np.log10(lum_low) + np.log10(lum_high)) / 2.0)
        bin_centers.append(bin_center)

        completeness = []
        completeness_err = []
        V = []
        for j in range(len(cat_)):
            f = cat_[col_flux][j]
            z = cat_[col_z][j]
            c = cat_[col_comp][j]
            c_err = cat_[col_comp_err][j]
            vol = v_max(f, flux_lim, z, area, survey_min, survey_max)
            V.append(vol.to_value(u.Mpc**3))
            completeness.append(c)
            completeness_err.append(c_err)
        
        V = np.array(V)
        completeness = np.array(completeness)
        completeness_err = np.array(completeness_err)
        
        if np.any(completeness <= 0):
            print(f"Warning: bin {i} includes completeness <= 0")
        
        valid = (completeness > 0) & (V > 0)
        n_valid = np.sum(valid)
        n_gal.append(n_valid)
        
        if n_valid < min_n:
            phi.append(np.nan)
            phi_err.append(np.nan)
            continue
        
        weights = 1 / (completeness[valid] * V[valid])

        rel_c_err = (completeness_err[valid] / completeness[valid])**2
        weights_var = weights**2 * (1. +  rel_c_err)
        
        Phi = np.sum(weights) / delta_log_lum
        Phi_err = np.sqrt(np.sum(weights_var)) / delta_log_lum

        phi.append(Phi)
        phi_err.append(Phi_err)

    return np.array(bin_centers), np.array(phi), np.array(phi_err), np.array(n_gal)



def completeness_function(x, A, B, C, D):
    return A * erf(B * (x - C )) + D

def fit_erf(x_data, y_data):
    popt, _ = curve_fit(completeness_function,
                           x_data, y_data,
                           p0=[0.5, 1e-42, 1e42, 0.5])
    
    return popt

def completeness_model(x, popt):
    return completeness_function(x, *popt)


def radec_z_to_cartesian(ra_deg, dec_deg, z, cosmology=cosmo):
    """RA/DEC [deg] + Redshift -> comoving kartesische Koordinaten [Mpc]."""
    d_c = cosmology.comoving_distance(z).to(u.Mpc).value 
    coords = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
    x = d_c * np.cos(coords.dec.rad) * np.cos(coords.ra.rad)
    y = d_c * np.cos(coords.dec.rad) * np.sin(coords.ra.rad)
    z_cart = d_c * np.sin(coords.dec.rad)
    return np.column_stack([x, y, z_cart])
 

def knn_density(positions, k=8, query_positions=None):
    """
    kNN-Dichteschätzer: rho = k / (4/3 * pi * r_k^3)
    r_k = Distanz zum k-ten Nachbarn (Galaxie selbst nicht mitgezählt).
    """
    tree = cKDTree(positions)
    if query_positions is None:
        query_positions = positions
        k_query = k + 1
        offset = 1
    else:
        k_query = k
        offset = 0
 
    dist, _ = tree.query(query_positions, k=k_query)
    r_k = dist[:, -1] 
    volume = (4.0 / 3.0) * np.pi * r_k**3
    density = k / volume
    return density, r_k