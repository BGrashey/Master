import os
import shutil
import numpy as np
import pandas as pd
import zarr
import dask
import dask.array as da
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from scipy.spatial import cKDTree

from tools.simulation import (
    build_fov_mask,
    generate_positions,
    inject_sources_into_cube,
)
from tools.fof_min import fof_minimal_zarr, catalog_to_wcs_table


def match_injected_to_detected(df_inj: pd.DataFrame, tab_det: Table, max_dist_spat_pix=3.0, max_dist_spec_pix=4.0):
    """
    Gleicht simulierte Positionen mit FoF-Detektionen in einem skalierten 3D-KDTree ab.
    Ergänzt das DataFrame um die Spalte 'detected' (1 = gefunden, 0 = nicht).
    """
    df_res = df_inj.copy()
    if len(tab_det) == 0:
        df_res["detected"] = 0
        return df_res

    det_df = tab_det.to_pandas()

    # Skalierungsfaktor für z, damit die Suchkugel im KDTree räumlich und spektral passt
    scale_z = max_dist_spat_pix / max_dist_spec_pix

    det_xyz = np.column_stack([
        det_df["ra_center"],
        det_df["dec_center"],
        det_df["wave_center"] * scale_z
    ])
    inj_xyz = np.column_stack([
        df_res["x_pix"],
        df_res["y_pix"],
        df_res["z_pix"] * scale_z
    ])

    tree = cKDTree(det_xyz)
    dists, _ = tree.query(inj_xyz, distance_upper_bound=max_dist_spat_pix)

    df_res["detected"] = (dists < max_dist_spat_pix).astype(int)
    return df_res


def run_completeness_pipeline():
    # =========================================================================
    # 1. KONFIGURATION & PFADE
    # =========================================================================
    base_dir = "/data/hetdex/u/bgrashey"
    input_fits = os.path.join(base_dir, "cubes/ssa22_fullfp_stack.fits")
    region_path = os.path.join(base_dir, "regions/fov.reg")
    real_cat_path = os.path.join(base_dir, "data_/combined_manual_vdfi_matched.fits")

    out_injected_fits = os.path.join(base_dir, "data_/injected_matched_results.fits")
    out_fof_catalog = os.path.join(base_dir, "data_/completeness_fof_catalog.fits")
    temp_zarr_path = os.path.join(base_dir, "cubes/temp_injected.zarr")

    # FoF- und Simulationsparameter
    sn_threshold = 2.
    linking_length = 2.0
    chunk_size_spec = 50
    n_per_bin = 50

    flux_bins = [
        (0.05, 0.10),   # Bereich unterhalb der Nachweisgrenze
        (0.10, 0.20),   # Einsetzen der Detektion
        (0.20, 0.40),   # Steiler Anstieg (Turnover)
        (0.40, 0.80),   # Hohe Completeness
        (0.80, 2.00),   # Plateau (~100%)
    ]

    # =========================================================================
    # 2. DATEN LADEN & TESTQUELLEN-POSITIONEN VORBEREITEN
    # =========================================================================
    print("-> Lade Flux-Cube und WCS...")
    with fits.open(input_fits, memmap=True) as hdul:
        header = hdul[0].header
        wcs = WCS(header)
        # Zunächst nur HDU 0 in den RAM laden, um Speicherpeaks zu vermeiden
        flux_cube = hdul[0].data.astype(np.float32)

    nz, ny, nx = flux_cube.shape

    print("-> Lade realen Katalog und erstelle FOV-Maske...")
    fits_table = Table.read(real_cat_path)
    real_catalog_world = [
        [float(row["ra_vdfi"]), float(row["dec_vdfi"]), float(row["z_vdfi"])]
        for row in fits_table
    ]

    fov_mask = build_fov_mask(nz, ny, nx, wcs, reg_file=region_path)
    chunks = [(i, min(i + chunk_size_spec, nz)) for i in range(0, nz, chunk_size_spec)]

    print("-> Generiere Positionen für Testgalaxien...")
    positions = generate_positions(
        nz, ny, nx, wcs,
        existing_catalog=real_catalog_world,
        chunks=chunks,
        n_bins=len(flux_bins),
        n_per_bin=n_per_bin,
        fov_mask=fov_mask,
    )

    # =========================================================================
    # 3. QUELLEN INJIZIEREN & S/N-WÜRFEL ERZEUGEN
    # =========================================================================
    print(f"-> Injiziere {len(positions)} Quellen...")
    flux_cube, new_sources_catalog = inject_sources_into_cube(
        flux_cube, positions, flux_bins
    )

    print("-> Lade Error-Cube und berechne S/N...")
    with fits.open(input_fits, memmap=True) as hdul:
        error_cube = hdul[1].data.astype(np.float32)

    mask_invalid = ~(error_cube > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        flux_cube /= error_cube
    flux_cube[mask_invalid] = np.nan

    # Sofortiger Speicherabwurf des Error-Arrays
    del error_cube, mask_invalid

    # =========================================================================
    # 4. ZARR SPEICHERN & DASK FOF AUSFÜHREN
    # =========================================================================
    print(f"-> Schreibe S/N-Cube nach Zarr ({temp_zarr_path})...")
    if os.path.exists(temp_zarr_path):
        shutil.rmtree(temp_zarr_path)

    zarr_chunk_shape = (min(50, nz), 200, 200)

    # Array explizit erstellen und befüllen
    z_arr = zarr.create_array(
        temp_zarr_path,
        shape=flux_cube.shape,
        chunks=zarr_chunk_shape,
        dtype=np.float32,
        overwrite=True,
    )
    z_arr[:] = flux_cube

    # Flux-Cube aus dem RAM entfernen, um Speicher für Dask freizumachen
    del flux_cube

    print("-> Initialisiere Dask-Array und berechne FoF-Graphen...")
    # Direkt das geöffnete Zarr-Objekt an Dask übergeben:
    sn_cube = da.from_zarr(z_arr)
    binary_mask = sn_cube > sn_threshold

    catalog_delayed = fof_minimal_zarr(binary_mask, linking_length=linking_length)

    print("-> Führe FoF aus (dask.compute)...")
    cat_computed = dask.compute(catalog_delayed)[0]

    final_fof_table = catalog_to_wcs_table(cat_computed, wcs_header=header)
    final_fof_table.write(out_fof_catalog, overwrite=True)
    print(f"-> FoF abgeschlossen: {len(final_fof_table)} Cluster gefunden.")

    # =========================================================================
    # 5. MATCHING & ERGEBNISTABELLE SPEICHERN
    # =========================================================================
    print("-> Führe 3D-Matching für Completeness durch...")
    df_injected = pd.DataFrame(new_sources_catalog)
    df_matched = match_injected_to_detected(df_injected, final_fof_table)

    # Beobachtete Wellenlänge (in Angström) als Auswerte-Feature ergänzen
    df_matched["wave_obs"] = 3470.0 + (df_matched["z_pix"] - 1.0) * 2.0

    tbl_out = Table.from_pandas(df_matched)
    tbl_out.write(out_injected_fits, overwrite=True)

    rec_count = int(df_matched["detected"].sum())
    total_count = len(df_matched)
    percentage = (rec_count / total_count * 100) if total_count > 0 else 0.0

    print("=" * 60)
    print(f"Fertig! Wiedergefunden: {rec_count} / {total_count} Quellen ({percentage:.2f}%)")
    print(f"Ergebnisse gespeichert unter: {out_injected_fits}")
    print(f"FoF-Katalog gespeichert unter: {out_fof_catalog}")
    print("=" * 60)

    # Temporäres Zarr-Verzeichnis optional entfernen:
    # shutil.rmtree(temp_zarr_path)


if __name__ == "__main__":
    run_completeness_pipeline()
