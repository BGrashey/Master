import math
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table

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
    num_cutouts     = 25,
    rest_wavelength = 1215.67,
    # Stretch
    use_fixed_stretch = True,
    vmin_fixed      = -2.0,
    vmax_fixed      =  1.0,
    cmap            = "inferno",
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
    #plt.savefig(output_file, dpi=300, bbox_inches="tight",
    #            facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig)
    print(f"  Gespeichert: {output_file}")

    
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u

def completeness_analysis(injected_path, detected_path, sky_radius=3.0, dz=0.5):
    inj = Table.read(injected_path)
    det = Table.read(detected_path)
    
    coords_inj = SkyCoord(ra=inj["ra"] * u.deg, dec=inj["dec"] * u.deg)
    coords_det = SkyCoord(ra=det["ra"] * u.deg, dec=det["dec"] * u.deg)
    
    idx, sep, _ = coords_inj.match_to_catalog_sky(coords_det)
    
    delta_z = np.abs(np.array(inj["z"]) - np.array(det["z"])[idx])
    
    inj["detected"] = (sep.arcsec <= sky_radius) & (delta_z <= dz)
    
    inj.write(injected_path.replace(".fits", "_flagged.fits"), overwrite=True)
    return inj