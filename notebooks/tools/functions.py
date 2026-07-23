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


COLNAMES = {
    "ra":  ["ra", "RA", "Ra", "RAJ2000", "ra_vdfi", "ra_hetdex"],
    "dec": ["dec", "DEC", "Dec", "DEJ2000", "dec_vdfi", "dec_hetdex"],
    "z":   ["z", "Z", "redshift", "REDSHIFT", "zspec", "ZSPEC", "z_vdfi", "z_hetdex", "redshift"],
    "flux":    ["flux", "Flux", "FLUX", "flux_lya"],
    "luminosity": ["lum", "luminosity", "luminosity_lae", "LUMINOSITY"],
    "completeness": ["completeness", "comp", "COMPLETENESS"],
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
    col_seg         = None,          # Optional, falls nicht vorhanden
    # Filter
    filters         = None,          # Liste von Dicts wie CATALOG_FILTERS
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
    label           = None,         # Spalte im Katalog mit bool werten
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

    # ── Hilfsfunktionen ───────────────────────────────────────────────────────

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
            return None
        x0, x1 = int(round(x)) - half, int(round(x)) - half + size
        y0, y1 = int(round(y)) - half, int(round(y)) - half + size
        if num_wave_slices >= 1:
            slice_img = cube_data[(z_slice-num_wave_slices):(z_slice+num_wave_slices), :, :]
        else:
            slice_img = cube_data[z_slice:z_slice+1, :, :]
        #slice_img = np.nanmean(slice_img, axis=0)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
            slice_img = np.nanmean(slice_img, axis=0)
        cutout = np.full((size, size), np.nan)
        sx0, sx1 = max(x0, 0), min(x1, nx)
        sy0, sy1 = max(y0, 0), min(y1, ny)
        cx0 = sx0 - x0
        cx1 = cx0 + (sx1 - sx0)
        cy0 = sy0 - y0
        cy1 = cy0 + (sy1 - sy0)
        if sx1 > sx0 and sy1 > sy0:
            cutout[cy0:cy1, cx0:cx1] = slice_img[sy0:sy1, sx0:sx1]
        return cutout.astype(float)

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

    # ── Katalog laden ─────────────────────────────────────────────────────────

    """print(f"\nLade Katalog: {catalog_file}")
    with fits.open(catalog_file) as hdul:
        cat = hdul[1].data
    print(f"  {len(cat)} Objekte geladen.")"""

    cat = catalog[:num_cutouts]
    
    col_ra = _find_col(cat, COLNAMES["ra"])
    col_dec = _find_col(cat, COLNAMES["dec"])
    col_z = _find_col(cat, COLNAMES["z"])

    # Filter anwenden
    if filters:
        print(f"  Wende {len(filters)} Filter an:")
        mask = apply_filters(cat, filters)
        cat  = cat[mask]
        print(f"  → {len(cat)} Objekte nach Filterung.")

    ra_arr  = cat[col_ra]
    dec_arr = cat[col_dec]
    z_arr   = cat[col_z]
    seg_arr = cat[col_seg] if col_seg else [None] * len(cat)

    # ── Cube laden ────────────────────────────────────────────────────────────

    print(f"Lade Cube: {cube_file}")
    with fits.open(cube_file, memmap=True) as hdul:
        cube_data = hdul[0].data
        wcs       = WCS(hdul[0].header)
    print(f"  Shape: {cube_data.shape}")

    # ── Grid erstellen ────────────────────────────────────────────────────────

    n_obj = len(ra_arr)
    ncols = cols_in_grid
    nrows = math.ceil(n_obj / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 2.8, nrows * 2.8),
        facecolor="#0f0f0f",
    )
    axes_flat = np.array(axes).reshape(-1)

    for ax in axes_flat:
        ax.set_facecolor("#0f0f0f")
        ax.axis("off")

    print(f"  Erzeuge {n_obj} Cutouts ...")

    for i, (ra, dec, z, seg) in enumerate(zip(ra_arr, dec_arr, z_arr, seg_arr)):
        ax = axes_flat[i]
        x_pix = y_pix = z_slice = cutout = None

        try:
            x_pix, y_pix = sky_to_pixel(ra, dec, wcs)
            z_slice = redshift_to_slice(z)
            cutout  = extract_cutout(cube_data, x_pix, y_pix, z_slice, cutout_size)
        except Exception as e:
            print(f"    Objekt {i+1}: Fehler – {e}")

        ax.axis("on")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#555555")
            spine.set_linewidth(0.6)

        if cutout is None or np.all(np.isnan(cutout)):
            ax.set_facecolor("#111111")
            ax.text(0.5, 0.5, "N/A", color="gray", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9)
        else:
            vmin = vmin_fixed if use_fixed_stretch else np.nanpercentile(cutout, 1)
            vmax = vmax_fixed if use_fixed_stretch else np.nanpercentile(cutout, 99)
            ax.imshow(cutout, origin="lower", cmap=cmap,
                      vmin=vmin, vmax=vmax, interpolation="nearest")
            if label and cat[label][i]:
                circle = plt.Circle((cutout_size/2, cutout_size/2), radius=3,
                         color="lime", fill=False, linewidth=1.2)
                ax.add_patch(circle)

        seg_str  = f"{int(seg)}" if seg is not None else "–"
        obs_wave = rest_wavelength * (1.0 + z)
        ax.set_title(
            f"#{i+1}  z={z:.3f}  seg={seg_str}\n{obs_wave:.0f} Å",
            color="white", fontsize=6.0, pad=2,
        )

        coord_label = (f"x={int(round(x_pix))}  y={int(round(y_pix))}  sl={z_slice}"
                       if x_pix is not None else "x=?  y=?  sl=?")
        ax.text(0.5, -0.04, coord_label, color="#aaaaaa",
                ha="center", va="top", transform=ax.transAxes,
                fontsize=5.5, fontfamily="monospace")

    stretch_note = (f"[stretch: {vmin_fixed} … {vmax_fixed}]"
                    if use_fixed_stretch else "[stretch: per-tile percentile]")
    fig.suptitle(
        f"{title}\nCutouts {cutout_size}×{cutout_size} px  |  "
        f"λ_rest={rest_wavelength:.2f} Å  |  {stretch_note}",
        color="white", fontsize=10, y=1.01,
    )
    plt.tight_layout(pad=0.5)
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


#-------------------
# Neue Funktion um Kataloge zu matchen
#-------------------

 
def _load(path):
    fmt = "fits" if path.endswith((".fits", ".fit")) else "csv"
    return Table.read(path, format=fmt)
 
def crossmatch(path1, path2, rad=2.5, dz_max=0.1, use_z=True):

    t1, t2 = _load(path1), _load(path2)

    cat1 = SkyCoord(
        ra=t1[_find_col(t1, COLNAMES["ra"])].data * u.deg,
        dec=t1[_find_col(t1, COLNAMES["dec"])].data * u.deg,
    )
    cat2 = SkyCoord(
        ra=t2[_find_col(t2, COLNAMES["ra"])].data * u.deg,
        dec=t2[_find_col(t2, COLNAMES["dec"])].data * u.deg,
    )

    i1, i2, _, _ = search_around_sky(cat1, cat2, rad * u.arcsec)

    if use_z:
        z1 = np.asarray(t1[_find_col(t1, COLNAMES["z"])])
        z2 = np.asarray(t2[_find_col(t2, COLNAMES["z"])])

        mask = np.abs(z1[i1] - z2[i2]) < dz_max
        matched_idx = np.unique(i1[mask])
    else:
        matched_idx = np.unique(i1)

    t1["MATCHED"] = np.zeros(len(t1), dtype=bool)
    t1["MATCHED"][matched_idx] = True

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
    
    


def z_max_fluxlim(flux_lim, flux, redshift, survey_max=3.4):
    if flux < flux_lim:
        z = redshift
    else:
        dl = np.sqrt(flux / flux_lim) * cosmo.luminosity_distance(redshift)
        z = z_at_value(cosmo.luminosity_distance, dl)
    
    return np.minimum(z, survey_max)



def v_max(flux, flux_lim, redshift, area=1000*u.deg**2, survey_min=2.7, survey_max=3.4):
    z_min = survey_min
    z_max = z_max_fluxlim(flux_lim, flux, redshift, survey_max)
    area_sr = area.to(u.sr).value
    
    v_max = cosmo.comoving_volume(z_max)
    v_min = cosmo.comoving_volume(z_min)

    volume = (v_max - v_min) * (area_sr / (4 * np.pi))

    return volume




def luminosity_function(flux_lim, catalog,
                        num_bins=10,
                        area=0.07056104808102222*u.deg**2,
                        survey_min=2.77,
                        survey_max=3.43):
    
    cat = Table.read(catalog)

    phi = []
    bin_centers = []

    col_z = _find_col(cat, COLNAMES["z"])
    col_flux = _find_col(cat, COLNAMES["flux"])
    col_lum = _find_col(cat, COLNAMES["luminosity"])
    col_comp = _find_col(cat, COLNAMES["completeness"])

    lum_low = np.log10(np.min(cat[col_lum]))
    lum_high = np.log10(np.max(cat[col_lum]))

    bin_edges = np.logspace(lum_low, lum_high, num_bins+1)

    for i in range(num_bins):
        lum_low = bin_edges[i]
        lum_high = bin_edges[i+1]
        
        delta_log_lum = np.log10(lum_high) - np.log10(lum_low)
        
        bin_center = 10**( (np.log10(lum_low) + np.log10(lum_high)) / 2.0 )
        bin_centers.append(bin_center)

        mask = (cat[col_lum] >= lum_low) & (cat[col_lum] < lum_high)
        cat_ = cat[mask]

        completeness = []
        V = []
        for j in range(len(cat_)):
            f = cat_[col_flux][j]
            z = cat_[col_z][j]
            c = cat_[col_comp][j]
            vol = v_max(f, flux_lim, z, area, survey_min, survey_max)
            V.append(vol.to_value(u.Mpc**3))
            completeness.append(c)
        
        V = np.array(V)
        completeness = np.array(completeness)
        
        Phi = np.sum(1 / (completeness * V)) / delta_log_lum

        phi.append(Phi)

    return np.array(bin_centers), np.array(phi)



def completeness_function(x, A, B, C, D):
    return A * erf(B * (x - C )) + D

def fit_erf(x_data, y_data):
    popt, _ = curve_fit(completeness_function,
                           x_data, y_data,
                           p0=[1, 1, 0, 0])
    
    return popt

def completeness_model(x, popt):
    return completeness_function(x, *popt)


def radec_z_to_cartesian(ra_deg, dec_deg, z, cosmology=cosmo):
    """RA/DEC [deg] + Redshift -> comoving kartesische Koordinaten [Mpc]."""
    d_c = cosmology.comoving_distance(z).to(u.Mpc).value  # comoving distance
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
    r_k = dist[:, -1]  # Distanz zum k-ten (echten) Nachbarn
    volume = (4.0 / 3.0) * np.pi * r_k**3
    density = k / volume
    return density, r_k