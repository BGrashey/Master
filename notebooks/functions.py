import math
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
import astropy.units as u
from astropy.coordinates import SkyCoord

def plot_cutout_grid(
    catalog,
    cube_file,
    output_file,
    # Spaltennamen
    col_ra          = "RA",
    col_dec         = "DEC",
    col_z           = "z",
    col_seg         = None,          # Optional, falls nicht vorhanden
    # Filter
    filters         = None,          # Liste von Dicts wie CATALOG_FILTERS
    # Grid-Optionen
    cutout_size     = 20,
    cols_in_grid    = 5,
    rest_wavelength = 1215.67,
    # Stretch
    use_fixed_stretch = True,
    vmin_fixed      = -2.0,
    vmax_fixed      =  1.0,
    cmap            = "inferno",
    label = None,                    # Spaltenname: dtype = bool
    # Titel
    title           = None,
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

    def redshift_to_slice(z, wcs, rest_wave_aa):
        obs_wave = rest_wave_aa * (1.0 + z)
        dummy_ra  = wcs.wcs.crval[0]
        dummy_dec = wcs.wcs.crval[1]
        try:
            px, py, pz = wcs.all_world2pix(dummy_ra, dummy_dec, obs_wave, 0)
            return int(round(float(pz)))
        except Exception:
            crpix = wcs.wcs.crpix[2] - 1
            crval = wcs.wcs.crval[2]
            cdelt = (wcs.wcs.cdelt[2] if wcs.wcs.cdelt[2] != 0
                     else wcs.wcs.cd[2, 2])
            return int(round(crpix + (obs_wave - crval) / cdelt))

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
        slice_img = cube_data[z_slice, :, :]
        cutout = np.full((size, size), np.nan)
        sx0, sx1 = max(x0, 0), min(x1, nx)
        sy0, sy1 = max(y0, 0), min(y1, ny)
        cx0 = sx0 - x0
        cx1 = cx0 + (sx1 - sx0)
        cy0 = sy0 - y0
        cy1 = cy0 + (sy1 - sy0)
        if sx1 > sx0 and sy1 > sy0:
            cutout[cy0:cy1, cx0:cx1] = slice_img[sy0:sy1, sx0:sx1]
        return cutout

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

    cat = catalog

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
    with fits.open(cube_file) as hdul:
        cube_data = hdul[0].data.astype(float)
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
            z_slice = redshift_to_slice(z, wcs, rest_wavelength)
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


def read_catalog(filename: str):
    dir = "/Users/bene/Desktop/mpe/fits/catalogs"
    tbl = Table.read(os.path.join(dir, filename))
    return tbl


def crossmatch_catalogs(
    primary_path: str,
    out_path: str,
    ref_path: str,
    ra_col_primary: str = "RA",
    dec_col_primary: str = "DEC",
    ra_col_ref: str = "ra_vdfi",
    dec_col_ref: str = "dec_vdfi",
    z_col_primary: str = "z",
    z_col_ref: str = "z_vdfi",
    max_sep_arcsec: float = 2.0,
    max_delta_z: float = 5.0,
    flag_col: str = "matched_in_reference",
) -> None:
    
    cat_primary = Table.read(primary_path)
    cat_reference = Table.read(ref_path)

    cat_primary[flag_col] = False

    coords_primary = SkyCoord(
        ra=cat_primary[ra_col_primary] * u.deg,
        dec=cat_primary[dec_col_primary] * u.deg,
    )
    coords_ref = SkyCoord(
        ra=cat_reference[ra_col_ref] * u.deg,
        dec=cat_reference[dec_col_ref] * u.deg,
    )

    idx, sep2d, _ = coords_ref.match_to_catalog_sky(coords_primary)

    dz = np.abs(cat_reference[z_col_ref] - cat_primary[z_col_primary][idx])
    match_mask = (sep2d < max_sep_arcsec * u.arcsec) & (dz < max_delta_z)

    valid_primary_idx = idx[match_mask]
    valid_sep = sep2d[match_mask]

    sort_order = np.argsort(valid_sep)
    sorted_primary_idx = valid_primary_idx[sort_order]

    _, unique_mask = np.unique(sorted_primary_idx, return_index=True)
    final_primary_idx = sorted_primary_idx[unique_mask]

    cat_primary[flag_col][final_primary_idx] = True

    print(f"Matches gefunden:            {int(np.sum(cat_primary[flag_col]))} / {len(cat_primary)}")
    print(f"Referenzquellen ohne Match:  {len(cat_reference) - int(np.sum(match_mask))}")
    cat_primary.write(out_path, overwrite=True)


def crossmatch_and_merge_catalogs(
    primary_path: str,
    out_path: str,
    ref_path: str,
    ra_col_primary: str = "RA",
    dec_col_primary: str = "DEC",
    ra_col_ref: str = "ra_vdfi",
    dec_col_ref: str = "dec_vdfi",
    z_col_primary: str = "z",
    z_col_ref: str = "z_vdfi",
    max_sep_arcsec: float = 2.0,
    max_delta_z: float = 5.0,
    flag_col: str = "matched_in_reference",
    ref_prefix: str = "ref_",
) -> None:
    cat_primary = Table.read(primary_path)
    cat_reference = Table.read(ref_path)

    cols_to_drop = [c for c in cat_primary.colnames
                    if c.startswith(ref_prefix) or c == flag_col]
    if cols_to_drop:
        cat_primary.remove_columns(cols_to_drop)

    cat_primary[flag_col] = False

    for col_name in cat_reference.colnames:
        new_col_name = f"{ref_prefix}{col_name}"
        dtype = cat_reference[col_name].dtype
        if np.issubdtype(dtype, np.floating):
            fill_value = np.nan
        elif np.issubdtype(dtype, np.integer):
            fill_value = -1
        else:
            fill_value = ""
        cat_primary[new_col_name] = np.full(len(cat_primary), fill_value, dtype=dtype)

    coords_primary = SkyCoord(
        ra=cat_primary[ra_col_primary] * u.deg,
        dec=cat_primary[dec_col_primary] * u.deg,
    )
    coords_ref = SkyCoord(
        ra=cat_reference[ra_col_ref] * u.deg,
        dec=cat_reference[dec_col_ref] * u.deg,
    )

    idx, sep2d, _ = coords_ref.match_to_catalog_sky(coords_primary)

    dz = np.abs(cat_reference[z_col_ref] - cat_primary[z_col_primary][idx])
    match_mask = (sep2d < max_sep_arcsec * u.arcsec) & (dz < max_delta_z)

    valid_ref_idx = np.where(match_mask)[0]
    valid_primary_idx = idx[match_mask]
    valid_sep = sep2d[match_mask]

    sort_order = np.argsort(valid_sep)
    sorted_primary_idx = valid_primary_idx[sort_order]
    sorted_ref_idx = valid_ref_idx[sort_order]

    _, unique_mask = np.unique(sorted_primary_idx, return_index=True)
    final_primary_idx = sorted_primary_idx[unique_mask]
    final_ref_idx = sorted_ref_idx[unique_mask]

    cat_primary[flag_col][final_primary_idx] = True

    for col_name in cat_reference.colnames:
        new_col_name = f"{ref_prefix}{col_name}"
        cat_primary[new_col_name][final_primary_idx] = cat_reference[col_name][final_ref_idx]

    num_matches = int(np.sum(cat_primary[flag_col]))
    print(f"Matches gefunden und gemerged: {num_matches} / {len(cat_primary)}")
    print(f"Referenzquellen ohne Match:    {len(cat_reference) - num_matches}")

    cat_primary.write(out_path, overwrite=True)