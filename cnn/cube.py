import os
import warnings
import numpy as np
import torch
import torch.nn as nn
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.spatial import cKDTree
import configparser
import zarr
import psutil
from torch.utils.data import Dataset, DataLoader

from cnn import LAEDetector3D

config = configparser.ConfigParser()
config.read("config.ini")

warnings.filterwarnings("ignore", category=UserWarning)

# ── Pfade ─────────────────────────────────────────────────────────────────────
CUBE_FILE        = "/data/hetdex/u/bgrashey/cubes/injected_new.zarr"
FITS_HEADER_FILE = "/data/hetdex/u/bgrashey/cubes/ssa22_fullfp_stack.fits"
MODEL_FILE       = "lae_model.pt"
OUTPUT_FILE      = "/data/hetdex/u/bgrashey/data_/cnn/kandidaten_cube_search.fits"
TRUE_CAT         = "/data/hetdex/u/bgrashey/data_/cnn/regions_tabelle.fits"

# ── Parameter ─────────────────────────────────────────────────────────────────
LYA_REST             = 1215.67
HALF_Z               = config.getint("TRAINING", "HALF_Z")
HALF_Y               = config.getint("TRAINING", "HALF_Y")
HALF_X               = config.getint("TRAINING", "HALF_X")
DROPOUT              = config.getfloat("CNN", "DROPOUT")
STRIDE               = config.getint("CNN", "STRIDE", fallback=5)   # ← war 3, jetzt 5
THRESHOLD            = config.getfloat("CNN", "THRESHOLD")
SN_THRESHOLD_COARSE  = 1.0
NMS_RADIUS           = 2 * STRIDE
MATCH_RADIUS         = 3.0
MATCH_DZ             = 0.1
TOTAL_SOURCES        = None

# CPU-Kerne: alle für PyTorch-Intra-op
N_SLOTS      = int(os.environ.get("NSLOTS", os.cpu_count() or 4))
NUM_WORKERS  = max(1, N_SLOTS // 2)
torch.set_num_threads(N_SLOTS)

print(f"STRIDE={STRIDE}  |  torch-Threads={N_SLOTS}  |  DataLoader-Workers={NUM_WORKERS}")


# ── PyTorch Dataset ───────────────────────────────────────────────────────────
class SlidingWindowDataset(Dataset):
    def __init__(self, cube_data, coords, dz, dy, dx):
        self.cube_data = cube_data
        self.coords    = coords
        self.dz, self.dy, self.dx = dz, dy, dx

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        z, y, x = self.coords[idx]
        dz, dy, dx = self.dz, self.dy, self.dx
        sub = np.nan_to_num(self.cube_data[
            z - dz : z + dz + 1,
            y - dy : y + dy + 1,
            x - dx : x + dx + 1
        ].astype(np.float32))
        return torch.tensor(sub).unsqueeze(0), torch.tensor([z, y, x])


# ── RAM-Check und Cube-Loading ────────────────────────────────────────────────
def load_cube(z_store):
    """
    Lädt den Cube chunk-weise in ein vorallokiertes float32-Array.
    Kein Zwischenpuffer im nativen Zarr-Dtype → kein OOM trotz großem Cube.
    """
    zarr_array   = z_store["PRIMARY"]
    cube_gb      = zarr_array.nbytes / 1e9
    needed_gb    = cube_gb * 2.2
    available_gb = psutil.virtual_memory().available / 1e9

    print(f"  Würfelgröße   : {cube_gb:.1f} GB")
    print(f"  Benötigt (Peak): {needed_gb:.1f} GB  |  Verfügbar: {available_gb:.1f} GB")

    if needed_gb < available_gb * 0.85:
        print("  → Lade Würfel chunk-weise in RAM (float32, kein Zwischenpuffer) ...")
        shape = zarr_array.shape
        cube_data = np.empty(shape, dtype=np.float32)

        chunks = zarr_array.chunks if hasattr(zarr_array, "chunks") else (64, 64, 64)
        cz, cy, cx = chunks

        total_chunks = (
            (shape[0] + cz - 1) // cz *
            (shape[1] + cy - 1) // cy *
            (shape[2] + cx - 1) // cx
        )
        done = 0

        for z0 in range(0, shape[0], cz):
            for y0 in range(0, shape[1], cy):
                for x0 in range(0, shape[2], cx):
                    z1 = min(z0 + cz, shape[0])
                    y1 = min(y0 + cy, shape[1])
                    x1 = min(x0 + cx, shape[2])
                    cube_data[z0:z1, y0:y1, x0:x1] = (
                        zarr_array[z0:z1, y0:y1, x0:x1].astype(np.float32)
                    )
                    done += 1
                    if done % max(1, total_chunks // 20) == 0:
                        print(f"    … {100*done/total_chunks:5.1f}%", end="\r", flush=True)

        print("\n  ✓ Würfel im RAM.")
        return cube_data, True
    else:
        print("  ⚠ Zu wenig RAM — bleibe bei lazy Zarr-Loading.")
        return zarr_array, False


# ── WCS-Hilfsfunktion ─────────────────────────────────────────────────────────
def get_wcs_axis_order(wcs: WCS):
    axis_names = [name.upper() for name in wcs.axis_type_names]
    print(f"  WCS-Achsen erkannt: {axis_names}")

    ra_ax   = next((i for i, n in enumerate(axis_names) if "RA"   in n), None)
    dec_ax  = next((i for i, n in enumerate(axis_names) if "DEC"  in n), None)
    wave_ax = next(
        (i for i, n in enumerate(axis_names)
         if any(k in n for k in ("WAVE", "LAMBDA", "FREQ", "VELO"))),
        None,
    )

    if None in (ra_ax, dec_ax, wave_ax):
        raise ValueError(
            f"Konnte RA/DEC/Wellenlängen-Achse nicht identifizieren. "
            f"Gefundene Achsen: {axis_names}"
        )
    return ra_ax, dec_ax, wave_ax


# ── GT-Diagnose ───────────────────────────────────────────────────────────────
def diagnose_gt_scores(model, cube_data, wcs, device, true_cat=TRUE_CAT):
    print("\n── GT-Diagnose: CNN-Score direkt an bekannten Quellen ──────────────")
    try:
        truth = Table.read(true_cat, format="fits")
    except Exception as e:
        print(f"  ✗ Konnte True-Katalog nicht laden: {e}")
        return

    ra_ax, dec_ax, wave_ax = get_wcs_axis_order(wcs)
    max_z, max_y, max_x = cube_data.shape
    dz, dy, dx = HALF_Z, HALF_Y, HALF_X

    scores = []
    for row in truth:
        ra, dec, redshift = float(row["RA"]), float(row["DEC"]), float(row["REDSHIFT"])
        wave_obs = LYA_REST * (1.0 + redshift)

        world = [None, None, None]
        world[ra_ax]   = ra
        world[dec_ax]  = dec
        world[wave_ax] = wave_obs

        try:
            pixel = wcs.all_world2pix([world], 0)[0]
        except Exception:
            continue

        px = int(round(float(pixel[ra_ax])))
        py = int(round(float(pixel[dec_ax])))
        pz = int(round(float(pixel[wave_ax])))

        if not (dz <= pz < max_z - dz and dy <= py < max_y - dy and dx <= px < max_x - dx):
            continue

        sub = np.nan_to_num(
            np.array(cube_data[pz-dz:pz+dz+1, py-dy:py+dy+1, px-dx:px+dx+1], dtype=np.float32)
        )

        with torch.no_grad():
            tensor = torch.tensor(sub, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            score  = torch.sigmoid(model(tensor)).item()

        scores.append(score)

    if scores:
        above = sum(s >= THRESHOLD for s in scores)
        print(f"\n  Zusammenfassung: {above}/{len(scores)} über Threshold ({THRESHOLD})")
        print(f"  Mittel={np.mean(scores):.3f}  Min={np.min(scores):.3f}  Max={np.max(scores):.3f}")
        print(f"  → Max. erreichbarer Recall beim Sliding Window: {above/len(scores):.1%}")
    print("─────────────────────────────────────────────────────────────────────\n")


# ── Optimierter Vorfilter: nur Stride-Gitter, vektorisiert ───────────────────
def filter_coords_by_sn_vectorized(cube_data, coords, sn_threshold):
    """
    Vorfilter NUR auf dem Stride-Gitter — kein maximum_filter über den
    gesamten Cube. Stattdessen:
      1. S/N-Werte an allen Gitterpunkten vektorisiert auslesen (ein Index-Zugriff)
      2. Lokales Maximum nur im Stride-Gitter prüfen (26er-Nachbarschaft)

    Physikalische Begründung: Das CNN hat nur auf zentrierte Quellen gelernt
    → wir müssen den Peak-Pixel des Stride-Gitters treffen, nicht den
    absoluten Sub-Pixel-Peak. Der maximum_filter über 5.8 Mrd. Pixel entfällt
    komplett.
    """
    coords_arr = np.array(coords, dtype=np.int32)
    zz, yy, xx = coords_arr[:, 0], coords_arr[:, 1], coords_arr[:, 2]

    # Alle S/N-Werte auf einmal auslesen (vektorisiert)
    vals = cube_data[zz, yy, xx]
    vals = np.where(np.isfinite(vals), vals, -np.inf)

    # 1) S/N-Vorfilter
    sn_mask = vals > sn_threshold
    n_after_sn = int(sn_mask.sum())
    print(f"  S/N>{sn_threshold}: {len(coords):,} → {n_after_sn:,} Punkte")

    if n_after_sn == 0:
        return []

    # 2) Lokales Maximum im Stride-Gitter
    #    Lookup-Dict: (z,y,x) → val, nur für Punkte die den S/N-Filter bestehen
    filtered_coords = coords_arr[sn_mask]
    filtered_vals   = vals[sn_mask]
    coord_to_val    = dict(zip(map(tuple, filtered_coords), filtered_vals))

    peak_coords = []
    for (z, y, x), v in coord_to_val.items():
        is_peak = True
        for dz in (-STRIDE, 0, STRIDE):
            for dy in (-STRIDE, 0, STRIDE):
                for dx in (-STRIDE, 0, STRIDE):
                    if dz == 0 and dy == 0 and dx == 0:
                        continue
                    if coord_to_val.get((z + dz, y + dy, x + dx), -np.inf) > v:
                        is_peak = False
                        break
                if not is_peak:
                    break
            if not is_peak:
                break
        if is_peak:
            peak_coords.append((z, y, x))

    print(f"  Peak-Filter   : {n_after_sn:,} → {len(peak_coords):,} Punkte "
          f"({100*len(peak_coords)/max(n_after_sn,1):.1f}% behalten)")
    return peak_coords


# ── Paralleles Sliding Window via DataLoader ──────────────────────────────────
def sliding_window_inference_parallel(model, cube_data, device, num_workers=NUM_WORKERS):
    max_z, max_y, max_x = cube_data.shape
    dz, dy, dx = HALF_Z, HALF_Y, HALF_X

    z_range = range(dz, max_z - dz, STRIDE)
    y_range = range(dy, max_y - dy, STRIDE)
    x_range = range(dx, max_x - dx, STRIDE)

    coords = [(z, y, x) for z in z_range for y in y_range for x in x_range]
    print(f"  {len(coords):,} Gitterpunkte auf Stride-Gitter.")

    coords = filter_coords_by_sn_vectorized(cube_data, coords, SN_THRESHOLD_COARSE)

    if not coords:
        print("  Keine Positionen nach Vorfilter übrig.")
        return [], [], [], []

    print(f"  Starte CNN-Inferenz auf {len(coords):,} Positionen ...")

    dataset    = SlidingWindowDataset(cube_data, coords, dz, dy, dx)
    dataloader = DataLoader(
        dataset,
        batch_size=512,        # kleiner als vorher: weniger Thread-Overhead auf CPU
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    found_z, found_y, found_x, probs_out = [], [], [], []
    model.eval()

    with torch.no_grad():
        for i, (subs_tensor, batch_coords_tensor) in enumerate(dataloader):
            subs_tensor = subs_tensor.to(device)
            probs = torch.sigmoid(model(subs_tensor)).cpu().numpy().flatten()
            batch_coords = batch_coords_tensor.numpy()

            for (z, y, x), prob in zip(batch_coords, probs):
                if prob > THRESHOLD:
                    found_z.append(int(z))
                    found_y.append(int(y))
                    found_x.append(int(x))
                    probs_out.append(float(prob))

            if i % 20 == 0:
                evaluated = min((i + 1) * dataloader.batch_size, len(coords))
                pct = 100 * evaluated / len(coords)
                print(f"  … {pct:5.1f}%  |  {len(found_x)} Treffer bisher", end="\r", flush=True)

    print()
    return found_z, found_y, found_x, probs_out


# ── Non-Maximum Suppression via KD-Tree ──────────────────────────────────────
def apply_nms(found_z, found_y, found_x, probs):
    if not found_x:
        print("  Keine Detektionen über dem Schwellenwert.")
        return [], [], [], []

    coords    = np.array([found_z, found_y, found_x], dtype=np.float32).T
    probs_arr = np.array(probs, dtype=np.float32)

    order     = np.argsort(probs_arr)[::-1]
    coords    = coords[order]
    probs_arr = probs_arr[order]

    kept = np.ones(len(coords), dtype=bool)
    tree = cKDTree(coords)

    for i in range(len(coords)):
        if not kept[i]:
            continue
        neighbors = tree.query_ball_point(coords[i], r=NMS_RADIUS)
        for j in neighbors:
            if j != i:
                kept[j] = False

    out_idx = np.where(kept)[0]
    print(f"  NMS: {len(found_x)} Rohdetektionen → {len(out_idx)} einzigartige Kandidaten.")

    return (
        coords[out_idx, 0].astype(int).tolist(),
        coords[out_idx, 1].astype(int).tolist(),
        coords[out_idx, 2].astype(int).tolist(),
        probs_arr[out_idx].tolist(),
    )


# ── Qualitätsevaluation ───────────────────────────────────────────────────────
def evaluate_candidates(candidates, true_cat_path, match_radius=MATCH_RADIUS,
                        match_dz=MATCH_DZ, total_sources=None):
    print("\n── Qualitätsevaluation ─────────────────────────────────────────────")

    try:
        truth = Table.read(true_cat_path, format="fits")
    except Exception as e:
        print(f"  ✗ Konnte True-Katalog nicht laden: {e}")
        return {}

    n_truth = total_sources if total_sources is not None else len(truth)
    n_cands = len(candidates)

    if n_cands == 0:
        print("  Keine Kandidaten zum Evaluieren.")
        return {}

    cand_coords  = SkyCoord(ra=candidates["RA"]  * u.deg, dec=candidates["DEC"] * u.deg)
    truth_coords = SkyCoord(ra=truth["RA"] * u.deg, dec=truth["DEC"] * u.deg)

    matched_truth = set()
    tp_indices    = []

    for i, (cand, cz) in enumerate(zip(cand_coords, candidates["REDSHIFT"])):
        sep   = cand.separation(truth_coords).arcsecond
        dz    = np.abs(truth["REDSHIFT"] - float(cz))
        match = np.where((sep <= match_radius) & (dz <= match_dz))[0]
        if len(match) > 0:
            best = match[np.argmin(sep[match])]
            matched_truth.add(int(best))
            tp_indices.append(i)

    tp = len(set(tp_indices))
    fp = n_cands - tp
    fn = n_truth - len(matched_truth)

    purity = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1     = (2 * purity * recall / (purity + recall) if (purity + recall) > 0 else 0.0)

    print(f"  Matching: ≤{match_radius}\" Winkelabstand, |Δz|≤{match_dz}")
    print(f"  Echte Quellen im Katalog : {n_truth}")
    print(f"  Gefundene Kandidaten     : {n_cands}")
    print()
    print(f"  True Positives   (TP)    : {tp}")
    print(f"  False Positives (FP)     : {fp}")
    print(f"  False Negatives (FN)     : {fn}")
    print()
    print(f"  Purity     (Precision)   : {purity:.3f}   ({tp}/{tp+fp})")
    print(f"  Completeness (Recall)    : {recall:.3f}   ({tp}/{tp+fn})")
    print(f"  F1-Score                 : {f1:.3f}")
    print("─────────────────────────────────────────────────────────────────")

    return {"tp": tp, "fp": fp, "fn": fn, "purity": purity, "recall": recall, "f1": f1}


# ── Haupt-Pipeline ────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Gerät: {device}")
    print(f"Nutze {NUM_WORKERS} DataLoader-Worker, {N_SLOTS} torch-Threads\n")

    print(f"[1/5] Lade Modell: {MODEL_FILE}")
    model = LAEDetector3D(dropout=DROPOUT)

    if torch.cuda.device_count() > 1:
        print(f"  --> Nutze {torch.cuda.device_count()} GPUs parallel via DataParallel!")
        model = nn.DataParallel(model)

    model = model.to(device)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    model.eval()

    # torch.compile beschleunigt Inferenz auf CPU um ~20-30% (PyTorch >= 2.0)
    #try:
    #    model = torch.compile(model)
    #    print("  ✓ torch.compile aktiviert.")
    #except Exception as e:
    #    print(f"  ⚠ torch.compile nicht verfügbar ({e}) — fahre ohne fort.")

    print("  Modell geladen.")

    print(f"\n[2/5] Lade IFU-Würfel: {CUBE_FILE}")
    z_store           = zarr.open_group(CUBE_FILE, mode='r')
    cube_data, in_ram = load_cube(z_store)
    wcs               = WCS(fits.getheader(FITS_HEADER_FILE))
    print(f"  Würfelgröße: {cube_data.shape}  (Z × Y × X)"
          f"  [{'RAM' if in_ram else 'lazy Zarr'}]")

    print(f"\n[3/5] GT-Diagnose vor dem Sliding Window …")
    diagnose_gt_scores(model, cube_data, wcs, device)

    print(f"[4/5] Sliding-Window (Stride={STRIDE}, S/N>{SN_THRESHOLD_COARSE}, "
          f"CNN-Schwelle={THRESHOLD}) …")
    found_z, found_y, found_x, probs = sliding_window_inference_parallel(
        model, cube_data, device, num_workers=NUM_WORKERS
    )
    print(f"  {len(found_x)} Pixel-Treffer über CNN-Schwellenwert {THRESHOLD}.")

    print(f"\n[4b/5] Non-Maximum Suppression via KD-Tree (Radius={NMS_RADIUS}px) …")
    found_z, found_y, found_x, probs = apply_nms(found_z, found_y, found_x, probs)

    if not found_x:
        print("✗ Keine Kandidaten nach NMS gefunden.")
        return

    ra_ax, dec_ax, wave_ax = get_wcs_axis_order(wcs)
    res_ra, res_dec, res_redshift = [], [], []

    for x, y_pix, z in zip(found_x, found_y, found_z):
        pixel = [None, None, None]
        pixel[ra_ax]   = x
        pixel[dec_ax]  = y_pix
        pixel[wave_ax] = z
        world = wcs.all_pix2world([pixel], 0)[0]
        res_ra.append(float(world[ra_ax]))
        res_dec.append(float(world[dec_ax]))
        wave_obs = float(world[wave_ax])
        res_redshift.append(wave_obs / LYA_REST - 1.0)

    t = Table(
        [res_ra, res_dec, res_redshift, probs],
        names=("RA", "DEC", "REDSHIFT", "Probability"),
    )
    t.sort("Probability")
    t.reverse()
    t.write(OUTPUT_FILE, format="fits", overwrite=True)
    print(f"\n✓ {len(t)} Kandidaten gespeichert: {OUTPUT_FILE}")

    print("\n[5/5] Evaluiere Kandidaten gegen Ground-Truth …")
    evaluate_candidates(t, TRUE_CAT, MATCH_RADIUS, MATCH_DZ, TOTAL_SOURCES)

    print(f"\n✓ Inference komplett abgeschlossen.")


if __name__ == "__main__":
    main()