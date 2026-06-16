"""
Halo Visualisierung
===================
Plot 1: Stack-Bilder nebeneinander (collapsed über spektrale Achse)
Plot 2: Radiale kpc-Profile — alle Quellen
Plot 3: Radiale kpc-Profile — Redshift-Bins
Plot 4: Radiale kpc-Profile — Dichte-Bins
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from astropy.io import fits
from pathlib import Path

# ──────────────────────────────────────────────
# KONFIGURATION — hier anpassen
# ──────────────────────────────────────────────

KPC_PER_PIX_OUT = 4.0       # muss mit halo_analysis.py übereinstimmen
N_DENSITY_BINS  = 4
N_Z_BINS        = 4

# Farbskala für Stack-Bilder (z.B. 'inferno', 'viridis', 'RdBu_r')
CMAP_IMAGE      = "inferno"
# Prozentile für Farbskalierung
VMIN_PCTL       = 1
VMAX_PCTL       = 99

OUTPUT_DIR      = Path(".")   # Wo die FITS-Stacks liegen und Plots gespeichert werden


# ──────────────────────────────────────────────
# HILFSFUNKTIONEN
# ──────────────────────────────────────────────

def load_stack(fname):
    """Lädt einen FITS-Stack. Gibt None zurück wenn Datei fehlt."""
    p = OUTPUT_DIR / fname
    if not p.exists():
        print(f"  [Warnung] Datei nicht gefunden: {p}")
        return None
    with fits.open(p) as hdul:
        return hdul[0].data.astype(float)


def collapse(stack):
    """Median über spektrale Achse (axis=0) → 2D-Bild."""
    return np.nanmedian(stack, axis=0)


def radial_profile(image, kpc_per_pix=KPC_PER_PIX_OUT, n_bins=20):
    """
    Berechnet das radiale Medienprofil eines 2D-Bildes.

    Returns
    -------
    r_kpc : 1D array — Radius in kpc (Bin-Mittelpunkte)
    profile : 1D array — Median-Flusswert pro Annulus
    """
    ny, nx = image.shape
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
    yy, xx = np.indices((ny, nx), dtype=float)
    r_pix  = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    r_kpc  = r_pix * kpc_per_pix

    r_max  = r_kpc.max()
    edges  = np.linspace(0, r_max, n_bins + 1)
    r_mid  = 0.5 * (edges[:-1] + edges[1:])

    profile = np.full(n_bins, np.nan)
    for i in range(n_bins):
        mask = (r_kpc >= edges[i]) & (r_kpc < edges[i + 1])
        vals = image[mask]
        if np.any(np.isfinite(vals)):
            profile[i] = np.nanmedian(vals)

    return r_mid, profile


# ──────────────────────────────────────────────
# DATEN LADEN
# ──────────────────────────────────────────────

# Stack: alle
stack_all = load_stack("stack_all.fits")

# Stacks: Dichte-Bins
density_stacks = []
density_labels = []
for b in range(1, N_DENSITY_BINS + 1):
    s = load_stack(f"stack_density_bin{b}_of{N_DENSITY_BINS}.fits")
    density_stacks.append(s)
    density_labels.append(f"Dichte-Bin {b}")

# Stacks: Redshift-Bins
z_stacks = []
z_labels = []
for b in range(1, N_Z_BINS + 1):
    s = load_stack(f"stack_zbin{b}_of{N_Z_BINS}.fits")
    z_stacks.append(s)
    z_labels.append(f"z-Bin {b}")


# ──────────────────────────────────────────────
# PLOT 1: Stack-Bilder nebeneinander
# ──────────────────────────────────────────────

all_stacks_flat = (
    [("Alle", stack_all)]
    + list(zip(density_labels, density_stacks))
    + list(zip(z_labels, z_stacks))
)
# Nur gültige Stacks
all_stacks_flat = [(lbl, s) for lbl, s in all_stacks_flat if s is not None]

n_plots = len(all_stacks_flat)
ncols   = min(n_plots, 5)
nrows   = int(np.ceil(n_plots / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.5 * nrows),
                         squeeze=False)
fig.suptitle("Median-Stacks (collapsed)", fontsize=14, y=1.01)

for idx, (label, stack) in enumerate(all_stacks_flat):
    ax   = axes[idx // ncols][idx % ncols]
    img  = collapse(stack)
    vmin = np.nanpercentile(img, VMIN_PCTL)
    vmax = np.nanpercentile(img, VMAX_PCTL)

    im = ax.imshow(img, origin="lower", cmap=CMAP_IMAGE,
                   vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(label, fontsize=10)

    # Achsenbeschriftung in kpc
    ny, nx = img.shape
    half_kpc = (nx // 2) * KPC_PER_PIX_OUT
    ticks_kpc = np.array([-1, -0.5, 0, 0.5, 1]) * half_kpc
    ticks_pix = ticks_kpc / KPC_PER_PIX_OUT + nx / 2
    ax.set_xticks(ticks_pix)
    ax.set_xticklabels([f"{t:.0f}" for t in ticks_kpc], fontsize=7)
    ax.set_yticks(ticks_pix)
    ax.set_yticklabels([f"{t:.0f}" for t in ticks_kpc], fontsize=7)
    ax.set_xlabel("kpc", fontsize=8)
    ax.set_ylabel("kpc", fontsize=8)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Leere Subplots ausblenden
for idx in range(len(all_stacks_flat), nrows * ncols):
    axes[idx // ncols][idx % ncols].set_visible(False)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "plot_stacks_images.pdf", bbox_inches="tight", dpi=150)
plt.savefig(OUTPUT_DIR / "plot_stacks_images.png", bbox_inches="tight", dpi=150)
print("→ plot_stacks_images.pdf/.png gespeichert")
plt.close()


# ──────────────────────────────────────────────
# PLOT-FUNKTION für radiale Profile
# ──────────────────────────────────────────────

colors_density = plt.cm.plasma(np.linspace(0.15, 0.85, N_DENSITY_BINS))
colors_z       = plt.cm.viridis(np.linspace(0.15, 0.85, N_Z_BINS))


def plot_profiles(stacks, labels, colors, title, fname):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_title(title, fontsize=13)

    for (label, stack), color in zip(zip(labels, stacks), colors):
        if stack is None:
            continue
        img = collapse(stack)
        r, prof = radial_profile(img)
        ax.plot(r, prof, marker="o", markersize=4, lw=1.8,
                label=label, color=color)

    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("Radius [kpc]", fontsize=12)
    ax.set_ylabel("Median Flux", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{fname}.pdf", bbox_inches="tight", dpi=150)
    plt.savefig(OUTPUT_DIR / f"{fname}.png", bbox_inches="tight", dpi=150)
    print(f"→ {fname}.pdf/.png gespeichert")
    plt.close()


# ──────────────────────────────────────────────
# PLOT 2: Radiales Profil — alle Quellen
# ──────────────────────────────────────────────

if stack_all is not None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_title("Radiales Profil — Alle Quellen", fontsize=13)
    img = collapse(stack_all)
    r, prof = radial_profile(img)
    ax.plot(r, prof, marker="o", markersize=5, lw=2, color="steelblue", label="Alle")
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("Radius [kpc]", fontsize=12)
    ax.set_ylabel("Median Flux", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plot_profile_all.pdf", bbox_inches="tight", dpi=150)
    plt.savefig(OUTPUT_DIR / "plot_profile_all.png", bbox_inches="tight", dpi=150)
    print("→ plot_profile_all.pdf/.png gespeichert")
    plt.close()


# ──────────────────────────────────────────────
# PLOT 3: Radiale Profile — Redshift-Bins
# ──────────────────────────────────────────────

plot_profiles(
    z_stacks, z_labels, colors_z,
    title="Radiale Profile — Redshift-Bins",
    fname="plot_profile_zbins",
)


# ──────────────────────────────────────────────
# PLOT 4: Radiale Profile — Dichte-Bins
# ──────────────────────────────────────────────

plot_profiles(
    density_stacks, density_labels, colors_density,
    title="Radiale Profile — Dichte-Bins",
    fname="plot_profile_densitybins",
)

print("\n✓ Alle Plots gespeichert.")