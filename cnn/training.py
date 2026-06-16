import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import (
    TensorDataset,
    DataLoader,
    random_split,
)

from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy.stats import mad_std

from sklearn.metrics import precision_recall_curve

import configparser

from cnn import LAEDetector3D

# ============================================================================
# CONFIG
# ============================================================================

config = configparser.ConfigParser()
config.read("config.ini")

warnings.filterwarnings("ignore", category=UserWarning)

CUBE_FILE   = "/data/hetdex/u/bgrashey/cubes/sn_cube_raw.fits"
TRUE_CAT    = "/data/hetdex/u/bgrashey/data_/cnn/regions_tabelle.fits"
FALSE_CAT   = "/data/hetdex/u/bgrashey/data_/cnn/false_examples.fits"

MODEL_FILE  = "lae_model.pt"
THRESH_FILE = "lae_threshold.txt"


LYA_REST = 1215.67
SEED     = 42

HALF_Z = config.getint("TRAINING", "HALF_Z")
HALF_Y = config.getint("TRAINING", "HALF_Y")
HALF_X = config.getint("TRAINING", "HALF_X")

BATCH_SIZE = 128
EPOCHS     = 50

# WICHTIG:
LR = 1e-4

DROPOUT     = config.getfloat("CNN", "DROPOUT")
VAL_SPLIT   = 0.15
TEST_SPLIT  = 0.10
PATIENCE    = 10
WEIGHT_DECAY = 1e-4

POS_WEIGHT = config.getfloat("TRAINING", "POS_WEIGHT")

N_HARD_NEGATIVES = config.getint("TRAINING", "N_HARD_NEG")

TARGET_PRECISION = config.getfloat("TRAINING", "PRECISION")

def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class AugmentDataset(torch.utils.data.Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):

        x, y = self.dataset[idx]

        # flip
        if np.random.rand() > 0.5:
            x = torch.flip(x, dims=[2])

        if np.random.rand() > 0.5:
            x = torch.flip(x, dims=[3])

        # rotation
        k = np.random.randint(0, 4)
        x = torch.rot90(x, k=k, dims=[2, 3])

        # multiplicative scaling
        if np.random.rand() > 0.5:
            scale = np.random.uniform(0.9, 1.1)
            x = x * scale

        # additive noise
        if np.random.rand() > 0.5:
            x = x + torch.randn_like(x) * 0.03

        return x, y

    def __len__(self):
        return len(self.dataset)

class FocalLoss(nn.Module):

    def __init__(self, gamma=2.0, pos_weight=1.0):

        super().__init__()

        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):

        bce = nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )

        probs = torch.sigmoid(logits)

        pt = torch.where(
            targets == 1,
            probs,
            1 - probs,
        )

        alpha = torch.where(
            targets == 1,
            torch.full_like(targets, self.pos_weight),
            torch.ones_like(targets),
        )

        loss = alpha * (1 - pt) ** self.gamma * bce

        return loss.mean()


class LAEDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv3d(
                1,
                16,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1),
            ),
            nn.InstanceNorm3d(
                16,
                affine=True,
            ),
            nn.GELU(),
            nn.Conv3d(
                16,
                32,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1),
            ),
            nn.InstanceNorm3d(
                32,
                affine=True,
            ),
            nn.GELU(),
            nn.AdaptiveMaxPool3d((None, 1, 1)),
        )
        self.spectral = nn.Sequential(

            nn.Conv1d(
                32,
                32,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),
            nn.Conv1d(
                32,
                64,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(32, 1),
        )
    def forward(self, x):
        x = self.spatial(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.spectral(x)
        x = self.head(x)
        return x


class LAEDetector3D(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.InstanceNorm3d(16, affine=True),
            nn.GELU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.InstanceNorm3d(32, affine=True),
            nn.GELU(),
            
            nn.AdaptiveMaxPool3d((1, 1, 1))
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(DROPOUT),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        return x

def get_wcs_axis_order(wcs):

    axis_names = [n.upper() for n in wcs.axis_type_names]

    print(f"  WCS-Achsen erkannt: {axis_names}")

    ra_ax = next(
        (i for i, n in enumerate(axis_names) if "RA" in n),
        None,
    )

    dec_ax = next(
        (i for i, n in enumerate(axis_names) if "DEC" in n),
        None,
    )

    wave_ax = next(
        (
            i
            for i, n in enumerate(axis_names)
            if any(k in n for k in ("WAVE", "LAMBDA"))
        ),
        None,
    )

    if None in (ra_ax, dec_ax, wave_ax):
        raise RuntimeError("WCS-Achsen nicht erkannt")

    return ra_ax, dec_ax, wave_ax

def extract_subcubes(
    cube_data,
    wcs,
    catalog_path,
    label,
):

    cat = Table.read(catalog_path)

    ra_ax, dec_ax, wave_ax = get_wcs_axis_order(wcs)

    max_z, max_y, max_x = cube_data.shape

    dz = HALF_Z
    dy = HALF_Y
    dx = HALF_X

    subs = []
    labels = []

    skipped = 0

    for row in cat:

        ra  = float(row["RA"])
        dec = float(row["DEC"])
        z   = float(row["REDSHIFT"])

        wave_obs = LYA_REST * (1 + z)

        world = [None, None, None]

        world[ra_ax]   = ra
        world[dec_ax]  = dec
        world[wave_ax] = wave_obs

        try:
            pixel = wcs.all_world2pix([world], 0)[0]

        except Exception:
            skipped += 1
            continue

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

        sub = cube_data[
            pz-dz:pz+dz+1,
            py-dy:py+dy+1,
            px-dx:px+dx+1,
        ].astype(np.float32)

        sub = np.nan_to_num(sub, nan=0.0)

        # WICHTIG:
        # KEIN median-subtraction mehr

        local_std = mad_std(sub) + 1e-6

        sub = sub / local_std

        subs.append(sub)
        labels.append(label)

    if skipped:
        print(f"  -> {skipped} Quellen übersprungen")

    X = np.expand_dims(
        np.array(subs, dtype=np.float32),
        axis=1,
    )

    y = np.array(labels, dtype=np.float32)

    return X, y

def sample_hard_negatives(
    cube_data,
    n_samples,
    seed=SEED,
):

    rng = np.random.default_rng(seed)

    max_z, max_y, max_x = cube_data.shape

    dz = HALF_Z
    dy = HALF_Y
    dx = HALF_X

    subs = []

    while len(subs) < n_samples:

        z = rng.integers(dz, max_z - dz)
        y = rng.integers(dy, max_y - dy)
        x = rng.integers(dx, max_x - dx)

        sub = cube_data[
            z-dz:z+dz+1,
            y-dy:y+dy+1,
            x-dx:x+dx+1,
        ].astype(np.float32)

        sub = np.nan_to_num(sub, nan=0.0)

        local_std = mad_std(sub) + 1e-6

        if local_std < 1e-6:
            continue

        sub = sub / local_std

        subs.append(sub)

    X = np.expand_dims(
        np.array(subs, dtype=np.float32),
        axis=1,
    )

    y = np.zeros(n_samples, dtype=np.float32)

    return X, y

def train_model(
    model,
    dl_train,
    dl_val,
    device,
):

    criterion = FocalLoss(
        gamma=2.0,
        pos_weight=POS_WEIGHT,
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
    )

    best_loss = np.inf
    best_state = None

    patience_counter = 0

    print(f"\n{'Epoche':>6} | {'Train':>10} | {'Val':>10}")
    print("-" * 36)

    for epoch in range(1, EPOCHS + 1):

        model.train()

        train_loss = 0.0

        for bx, by in dl_train:

            bx = bx.to(device)
            by = by.to(device).unsqueeze(1)

            optimizer.zero_grad()

            logits = model(bx)

            loss = criterion(logits, by)

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(dl_train)

        model.eval()

        val_loss = 0.0

        all_probs = []

        with torch.no_grad():

            for bx, by in dl_val:

                bx = bx.to(device)
                by = by.to(device).unsqueeze(1)

                logits = model(bx)

                loss = criterion(logits, by)

                val_loss += loss.item()

                probs = torch.sigmoid(logits)

                all_probs.extend(
                    probs.cpu().numpy().flatten()
                )

        val_loss /= len(dl_val)

        scheduler.step()

        prob_std = np.std(all_probs)

        print(
            f"{epoch:>6} | "
            f"{train_loss:>10.4f} | "
            f"{val_loss:>10.4f} | "
            f"prob_std={prob_std:.4f}"
        )

        if val_loss < best_loss:

            best_loss = val_loss

            best_state = {
                k: v.cpu().clone()
                for k, v in model.state_dict().items()
            }

            patience_counter = 0

        else:

            patience_counter += 1

            if patience_counter >= PATIENCE:

                print("\nEarly stopping")

                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model

def evaluate_model(
    model,
    dl,
    device,
    threshold=None,
):

    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():

        for bx, by in dl:

            probs = torch.sigmoid(
                model(bx.to(device))
            ).cpu().numpy().flatten()

            all_probs.extend(probs)

            all_labels.extend(
                by.numpy().astype(int)
            )

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    if threshold is None:

        prec, rec, thresholds = precision_recall_curve(
            all_labels,
            all_probs,
        )

        idx = np.where(
            prec[:-1] >= TARGET_PRECISION
        )[0]

        if len(idx) > 0:

            best_idx = idx[np.argmax(rec[idx])]

            threshold = float(thresholds[best_idx])

            print(f"\nThreshold: {threshold:.3f}")
            print(f"Precision: {prec[best_idx]:.3f}")
            print(f"Recall   : {rec[best_idx]:.3f}")

        else:

            threshold = 0.5

    preds = (all_probs >= threshold).astype(int)

    tp = ((preds == 1) & (all_labels == 1)).sum()
    fp = ((preds == 1) & (all_labels == 0)).sum()
    tn = ((preds == 0) & (all_labels == 0)).sum()
    fn = ((preds == 0) & (all_labels == 1)).sum()

    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)

    f1 = (
        2 * precision * recall /
        (precision + recall + 1e-9)
    )

    print("\nEvaluation")
    print("-" * 30)

    print(f"Precision : {precision:.3f}")
    print(f"Recall    : {recall:.3f}")
    print(f"F1        : {f1:.3f}")

    print("\nConfusion Matrix")
    print(f"TP={tp}  FP={fp}")
    print(f"FN={fn}  TN={tn}")

    return threshold

def main():

    set_seed()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Gerät: {device}")

    print("\n[1/5] Lade Cube")

    with fits.open(CUBE_FILE, memmap=False) as hdul:

        data_hdu = next(
            (
                h for h in hdul
                if h.data is not None and h.data.ndim == 3
            ),
            hdul[0],
        )

        cube_data = np.array(
            data_hdu.data,
            dtype=np.float32,
        )

        cube_data = np.nan_to_num(
            cube_data,
            nan=0.0,
        )

        wcs = WCS(data_hdu.header)

    print(f"Cube Shape: {cube_data.shape}")

    print("\n[2/5] Positive Samples")

    X_pos, y_pos = extract_subcubes(
        cube_data,
        wcs,
        TRUE_CAT,
        1.0,
    )

    print(f"Positive: {len(y_pos)}")

    print("\n[3/5] Hard Negatives")

    X_neg, y_neg = sample_hard_negatives(
        cube_data,
        N_HARD_NEGATIVES,
    )

    print(f"Negative: {len(y_neg)}")

    del cube_data

    X = np.concatenate([X_pos, X_neg])
    y = np.concatenate([y_pos, y_neg])

    dataset = TensorDataset(
        torch.tensor(X),
        torch.tensor(y),
    )

    val_size = int(len(dataset) * VAL_SPLIT)
    test_size = int(len(dataset) * TEST_SPLIT)

    train_size = len(dataset) - val_size - test_size

    ds_train, ds_val, ds_test = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    ds_train = AugmentDataset(ds_train)

    dl_train = DataLoader(
        ds_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    dl_val = DataLoader(
        ds_val,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    dl_test = DataLoader(
        ds_test,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    print(f"\nTrain: {len(ds_train)}")
    print(f"Val  : {len(ds_val)}")
    print(f"Test : {len(ds_test)}")

    print("\n[4/5] Trainiere Modell")

    model = LAEDetector3D(dropout=DROPOUT).to(device)

    model = train_model(
        model,
        dl_train,
        dl_val,
        device,
    )

    torch.save(
        model.state_dict(),
        MODEL_FILE,
    )

    print(f"\nModell gespeichert: {MODEL_FILE}")

    print("\n[5/5] Evaluation")

    threshold = evaluate_model(
        model,
        dl_val,
        device,
    )

    with open(THRESH_FILE, "w") as f:
        f.write(str(threshold))

    print(f"\nThreshold gespeichert: {threshold:.3f}")

    print("\nTESTSET")

    evaluate_model(
        model,
        dl_test,
        device,
        threshold=threshold,
    )

    print("done")

if __name__ == "__main__":
    main()