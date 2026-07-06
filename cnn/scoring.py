import warnings
import numpy as np
import torch
import torch.nn as nn

from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table

import configparser
import zarr

from cnn import LAEDetector3D


config = configparser.ConfigParser()
config.read("config.ini")

warnings.filterwarnings("ignore", category=UserWarning)

#ZARR_FILE    = "/data/hetdex/u/bgrashey/cubes/injected_new.zarr"
ZARR_FILE = "/data/hetdex/u/bgrashey/cubes/ssa22_fullfp_sig_filter.zarr"
FITS_HEADER  = "/data/hetdex/u/bgrashey/cubes/ssa22_fullfp_stack.fits"
MODEL_FILE   = "lae_model.pt"

#INPUT_CATALOG  = "/data/hetdex/u/bgrashey/data_/injected_sources_log_bins.fits"
INPUT_CATALOG = "/data/hetdex/u/bgrashey/data_/combined_manual_vdfi_rfscored.fits"
OUTPUT_CATALOG = "/data/hetdex/u/bgrashey/data_/combined_manual_vdfi_rf_cnn_scored.fits"

COL_RA       = "ra"
COL_DEC      = "dec"
COL_REDSHIFT = "z"
COL_OUTPUT   = "CNN_prob"

LYA_REST = 1215.67
HALF_Z = config.getint("TRAINING", "HALF_Z")
HALF_Y = config.getint("TRAINING", "HALF_Y")
HALF_X = config.getint("TRAINING", "HALF_X")
DROPOUT = config.getfloat("CNN", "DROPOUT")

DEBUG = False


def get_wcs_axis_order(wcs):
    axis_names = [n.upper() for n in wcs.axis_type_names]

    ra_ax = next(
        (i for i, n in enumerate(axis_names) if "RA" in n),
        None,
    )
    dec_ax = next(
        (i for i, n in enumerate(axis_names) if "DEC" in n),
        None,
    )
    wave_ax = next(
        (i for i, n in enumerate(axis_names)
         if any(k in n for k in ("WAVE", "LAMBDA"))),
        None,
    )

    if None in (ra_ax, dec_ax, wave_ax):
        raise RuntimeError(f"WCS axes not found: {axis_names}")

    return ra_ax, dec_ax, wave_ax


def normalize_subcube(sub):
    # Keine lokale Normalisierung — Modell wurde auf S/N-Daten trainiert
    return np.nan_to_num(sub.astype(np.float32), nan=0.0)


def score_with_tta(model, subcube, device):
    tensor = torch.tensor(
        subcube,
        dtype=torch.float32,
    ).unsqueeze(0).unsqueeze(0)

    scores = []

    with torch.no_grad():
        for k in range(4):
            t = torch.rot90(tensor, k=k, dims=[3, 4])
            score = torch.sigmoid(model(t.to(device))).item()
            scores.append(score)

            t_flip = torch.flip(t, dims=[3])
            score_flip = torch.sigmoid(model(t_flip.to(device))).item()
            scores.append(score_flip)

    return float(np.mean(scores))


def score_catalog(input_catalog, output_path):
    device = torch.device("cpu")
    print(f"Device: {device}\n")

    print(f"Loading model: {MODEL_FILE}")
    model = LAEDetector3D(dropout=DROPOUT).to(device)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    model.eval()

    print(f"Loading cube (lazy): {ZARR_FILE}")
    cube_data   = zarr.open(ZARR_FILE, mode='r')
    #cube_data = cube_["PRIMARY"]
    wcs       = WCS(fits.getheader(FITS_HEADER))
    print(f"Cube shape: {cube_data.shape}")

    ra_ax, dec_ax, wave_ax = get_wcs_axis_order(wcs)
    print(f"WCS axis order: ra_ax={ra_ax}, dec_ax={dec_ax}, wave_ax={wave_ax}")
    print(f"WCS axis names: {wcs.axis_type_names}")

    max_z, max_y, max_x = cube_data.shape
    dz, dy, dx = HALF_Z, HALF_Y, HALF_X

    print(f"\nScoring catalog: {input_catalog}")
    cat = Table.read(input_catalog)
    n_sources = len(cat)
    probabilities = np.full(n_sources, np.nan, dtype=np.float32)
    skipped = 0

    if DEBUG:
        row = cat[0]
        ra       = float(row[COL_RA])
        dec      = float(row[COL_DEC])
        z        = float(row[COL_REDSHIFT])
        wave_obs = LYA_REST * (1 + z)
        world    = [None, None, None]
        world[ra_ax]   = ra
        world[dec_ax]  = dec
        world[wave_ax] = wave_obs
        pixel = wcs.all_world2pix([world], 0)[0]
        px = int(round(pixel[ra_ax]))
        py = int(round(pixel[dec_ax]))
        pz = int(round(pixel[wave_ax]))
        print(f"\n── Debug erste Quelle ──")
        print(f"RA={ra:.4f}, DEC={dec:.4f}, z={z:.4f}")
        print(f"wave_obs={wave_obs:.2f}")
        print(f"pixel coords: px={px}, py={py}, pz={pz}")
        print(f"Cube shape (z,y,x): {max_z}, {max_y}, {max_x}")
        print(f"  z: {dz} <= {pz} < {max_z - dz}  -> {dz <= pz < max_z - dz}")
        print(f"  y: {dy} <= {py} < {max_y - dy}  -> {dy <= py < max_y - dy}")
        print(f"  x: {dx} <= {px} < {max_x - dx}  -> {dx <= px < max_x - dx}")
        print(f"────────────────────────\n")

    for i, row in enumerate(cat):
        if (i + 1) % max(1, n_sources // 20) == 0:
            print(
                f"Progress: {i+1}/{n_sources} ({100*(i+1)/n_sources:.0f}%)",
                end="\r",
            )

        try:
            ra       = float(row[COL_RA])
            dec      = float(row[COL_DEC])
            z        = float(row[COL_REDSHIFT])
            wave_obs = LYA_REST * (1 + z)

            world = [None, None, None]
            world[ra_ax]   = ra
            world[dec_ax]  = dec
            world[wave_ax] = wave_obs

            pixel = wcs.all_world2pix([world], 0)[0]

            px = int(round(pixel[ra_ax]))
            py = int(round(pixel[dec_ax]))
            pz = int(round(pixel[wave_ax]))

            if not (
                dz <= pz < max_z - dz and
                dy <= py < max_y - dy and
                dx <= px < max_x - dx
            ):
                skipped += 1
                continue

            sub = np.array(cube_data[
                pz-dz:pz+dz+1,
                py-dy:py+dy+1,
                px-dx:px+dx+1,
            ], dtype=np.float32)

            sub = normalize_subcube(sub)
            score = score_with_tta(model, sub, device)
            probabilities[i] = score

        except Exception as e:
            print(f"Source {i} ({ra:.4f}, {dec:.4f}, z={z:.4f}): {type(e).__name__}: {e}")
            skipped += 1
            continue

    print(" " * 80, end="\r")

    cat[COL_OUTPUT] = probabilities
    cat.sort(COL_OUTPUT)
    cat.reverse()
    cat.write(output_path, overwrite=True)

    evaluated = np.sum(~np.isnan(probabilities))
    print(f"\nEvaluated : {evaluated}/{n_sources}")
    print(f"Skipped   : {skipped}")
    print(f"Saved to  : {output_path}")


if __name__ == "__main__":
    score_catalog(
        input_catalog=INPUT_CATALOG,
        output_path=OUTPUT_CATALOG,
    )
