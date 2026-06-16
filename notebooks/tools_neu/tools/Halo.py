"""
Halo Analysis Pipeline
======================
- Berechnet lokale Galaxiendichte (kNN)
- Erstellt einen Dichte-Cube (3D)
- Stackt Lyman-Alpha-Emitter:
    1. Alle zusammen
    2. Nach Dichte-Bins (4 Quartile)
    3. Nach Redshift-Bins (4 gleich große Bins)
"""

import numpy as np
from astropy.io import fits
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
from scipy.spatial import cKDTree
from scipy.ndimage import map_coordinates
from tqdm import tqdm


# ──────────────────────────────────────────────
# KONFIGURATION — hier anpassen
# ──────────────────────────────────────────────

CUBE_FILE       = "cube.fits"          # Pfad zur FITS-Datei des Datenwürfels
CUBE_EXT        = 1                    # FITS-Extension des Würfels (oft 1 oder 0)
CATALOG_FILE    = "catalog.fits"       # Pfad zum Quellkatalog
CATALOG_EXT     = 1                    # FITS-Extension der Tabelle

# Spaltennamen im Katalog
COL_RA          = "RA"
COL_DEC         = "DEC"
COL_Z           = "Z"
COL_WAVE        = "WAVE"               # Beobachtete Wellenlänge in Ångström (Ly-alpha)
COL_X           = "X"                  # Pixelposition x im Würfel
COL_Y           = "Y"                  # Pixelposition y im Würfel
COL_ZPIX        = "ZPIX"              # Spektraler Pixelindex im Würfel

PIXSCALE_ARCSEC = 0.2                  # Pixelskala des Würfels [arcsec/pixel]
REST_WAVE       = 1216.0               # Ly-alpha Ruhewellenlänge [Å]
KPC_PER_PIX_OUT = 4.0                  # Gewünschte physikalische Auflösung [kpc/pixel]

STACK_SIZE      = 21                   # Räumliche Größe der Ausschnitte [pixel]
STACK_DEPTH     = 11                   # Spektrale Tiefe der Ausschnitte [planes]
N_DENSITY_BINS  = 4                    # Anzahl Dichte-Bins
N_Z_BINS        = 4                    # Anzahl Redshift-Bins
KNN_K           = 10                   # k für kNN-Dichte


# ──────────────────────────────────────────────
# 1. HILFSFUNKTIONEN (aus Halo.ipynb)
# ──────────────────────────────────────────────

def sky_to_comoving_xyz(ra_deg, dec_deg, redshift):
    ra  = np.deg2rad(np.asarray(ra_deg))
    dec = np.deg2rad(np.asarray(dec_deg))
    z   = np.asarray(redshift)
    Dc  = cosmo.comoving_distance(z).to_value(u.Mpc)
    x   = Dc * np.cos(dec) * np.cos(ra)
    y   = Dc * np.cos(dec) * np.sin(ra)
    zc  = Dc * np.sin(dec)
    return np.column_stack([x, y, zc])


def local_density_knn(ra_deg, dec_deg, redshift, k=KNN_K):
    pts = sky_to_comoving_xyz(ra_deg, dec_deg, redshift)
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=k + 1)
    rk = dists[:, k]
    volume = (4.0 / 3.0) * np.pi * rk**3
    rho = k / volume
    return rho, rk


def kpc_per_arcsec_from_wavelength(wavelength_angstrom, rest_wavelength=REST_WAVE):
    z = float(wavelength_angstrom) / float(rest_wavelength) - 1.0
    if not np.isfinite(z) or z < 0:
        raise ValueError(f"Ungültige Rotverschiebung z={z:.4f} für λ={wavelength_angstrom} Å")
    kpc_per_arcsec = cosmo.kpc_proper_per_arcmin(z).to(u.kpc / u.arcsec).value
    return z, float(kpc_per_arcsec)


def _nan_map_coordinates_2d(img, y_coords, x_coords, order=1, cval=np.nan):
    valid  = np.isfinite(img).astype(float)
    filled = np.nan_to_num(img, nan=0.0)
    coords = np.array([y_coords, x_coords])
    vals = map_coordinates(filled, coords, order=order, mode="constant", cval=0.0, prefilter=(order > 1))
    wgt  = map_coordinates(valid,  coords, order=1,     mode="constant", cval=0.0, prefilter=False)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = vals / wgt
    out[wgt < 1e-6] = cval
    return out


def extract_centered_cutout_fixed_kpc(
    cube, x, y, zpix, wavelength_angstrom,
    pixscale_arcsec=PIXSCALE_ARCSEC,
    size_out=STACK_SIZE,
    depth=STACK_DEPTH,
    kpc_per_pix_out=KPC_PER_PIX_OUT,
    order_zoom=1,
    rest_wavelength=REST_WAVE,
    treat_zeros_as_nan=True,
):
    nz, ny, nx = cube.shape
    z0   = int(np.round(zpix))
    zmin = z0 - depth // 2
    zmax = z0 + depth // 2 + 1
    if zmin < 0 or zmax > nz:
        return None, None

    _, kpc_per_arcsec = kpc_per_arcsec_from_wavelength(wavelength_angstrom, rest_wavelength)
    kpc_per_pix_in = float(pixscale_arcsec) * float(kpc_per_arcsec)
    zoom_factor    = kpc_per_pix_in / float(kpc_per_pix_out)
    if zoom_factor <= 0 or not np.isfinite(zoom_factor):
        return None, None

    c          = (size_out - 1) / 2.0
    margin     = 3 if order_zoom > 1 else 2
    half_in    = int(np.ceil(c / zoom_factor)) + margin
    x0, y0     = int(np.floor(x)), int(np.floor(y))
    xmin, xmax = x0 - half_in, x0 + half_in + 1
    ymin, ymax = y0 - half_in, y0 + half_in + 1
    if xmin < 0 or ymin < 0 or xmax > nx or ymax > ny:
        return None, None

    x_src_local = x - xmin
    y_src_local = y - ymin
    yy, xx      = np.indices((size_out, size_out), dtype=float)
    x_in = x_src_local + (xx - c) / zoom_factor
    y_in = y_src_local + (yy - c) / zoom_factor

    planes = []
    for zz in range(zmin, zmax):
        img = cube[zz, ymin:ymax, xmin:xmax].astype(float, copy=False)
        if treat_zeros_as_nan and np.any(img == 0.0):
            img = img.copy()
            img[img == 0.0] = np.nan
        planes.append(_nan_map_coordinates_2d(img, y_in, x_in, order=order_zoom))

    return np.asarray(planes, dtype=float), zoom_factor


# ──────────────────────────────────────────────
# 2. DATEN LADEN
# ──────────────────────────────────────────────

print("Lade Datenwürfel …")
with fits.open(CUBE_FILE) as hdul:
    cube   = hdul[CUBE_EXT].data.astype(float)
    header = hdul[CUBE_EXT].header
print(f"  Würfel-Shape: {cube.shape}  (nz, ny, nx)")

print("Lade Katalog …")
with fits.open(CATALOG_FILE) as hdul:
    cat = hdul[CATALOG_EXT].data
ra    = np.asarray(cat[COL_RA],   dtype=float)
dec   = np.asarray(cat[COL_DEC],  dtype=float)
zred  = np.asarray(cat[COL_Z],    dtype=float)
wave  = np.asarray(cat[COL_WAVE], dtype=float)
x_pix = np.asarray(cat[COL_X],   dtype=float)
y_pix = np.asarray(cat[COL_Y],   dtype=float)
z_pix = np.asarray(cat[COL_ZPIX],dtype=float)
N     = len(ra)
print(f"  {N} Quellen geladen.")


# ──────────────────────────────────────────────
# 3. LOKALE DICHTE BERECHNEN
# ──────────────────────────────────────────────

print(f"\nBerechne lokale Dichte (kNN, k={KNN_K}) …")
density, rk = local_density_knn(ra, dec, zred, k=KNN_K)
print(f"  Dichte-Bereich: {density.min():.4f} – {density.max():.4f} Gal/Mpc³")


# ──────────────────────────────────────────────
# 4. DICHTE-CUBE ERSTELLEN
# ──────────────────────────────────────────────
# Jede Quelle wird als Gaussian-Blob in den Würfel eingetragen.
# Alternativ: einfache Nearest-Pixel-Zuweisung.

print("\nErstelle Dichte-Cube …")
density_cube = np.zeros(cube.shape, dtype=float)
nz, ny, nx   = cube.shape

valid_mask = (
    np.isfinite(x_pix) & np.isfinite(y_pix) & np.isfinite(z_pix) &
    (x_pix >= 0) & (x_pix < nx) &
    (y_pix >= 0) & (y_pix < ny) &
    (z_pix >= 0) & (z_pix < nz)
)
xi = np.round(x_pix[valid_mask]).astype(int)
yi = np.round(y_pix[valid_mask]).astype(int)
zi = np.round(z_pix[valid_mask]).astype(int)
d  = density[valid_mask]

np.add.at(density_cube, (zi, yi, xi), d)

# Speichern
fits.writeto("density_cube.fits", density_cube, overwrite=True)
print("  → density_cube.fits gespeichert.")


# ──────────────────────────────────────────────
# 5. STACKING-FUNKTION
# ──────────────────────────────────────────────

def run_stack(indices, label):
    cutouts = []
    skipped = 0
    for i in tqdm(indices, desc=f"Stack: {label}"):
        sub, _ = extract_centered_cutout_fixed_kpc(
            cube, x_pix[i], y_pix[i], z_pix[i],
            wavelength_angstrom=wave[i],
        )
        if sub is not None:
            cutouts.append(sub)
        else:
            skipped += 1

    if len(cutouts) == 0:
        print(f"  [{label}] Keine gültigen Ausschnitte!")
        return None

    stack = np.nanmedian(np.array(cutouts), axis=0)
    fname = f"stack_{label}.fits"
    fits.writeto(fname, stack, overwrite=True)
    print(f"  [{label}] {len(cutouts)} Quellen gestackt ({skipped} übersprungen) → {fname}")
    return stack


# ──────────────────────────────────────────────
# 6. STACK 1: ALLE QUELLEN
# ──────────────────────────────────────────────

print("\n── Stack 1: Alle Quellen ──")
all_idx = np.arange(N)
run_stack(all_idx, "all")


# ──────────────────────────────────────────────
# 7. STACK 2: NACH DICHTE-BINS (Quartile)
# ──────────────────────────────────────────────

print(f"\n── Stack 2: {N_DENSITY_BINS} Dichte-Bins ──")
density_bins = np.quantile(density, np.linspace(0, 1, N_DENSITY_BINS + 1))
print(f"  Dichte-Grenzen: {np.round(density_bins, 4)}")

for b in range(N_DENSITY_BINS):
    lo, hi = density_bins[b], density_bins[b + 1]
    if b == N_DENSITY_BINS - 1:
        mask = (density >= lo) & (density <= hi)
    else:
        mask = (density >= lo) & (density < hi)
    idx = np.where(mask)[0]
    label = f"density_bin{b+1}_of{N_DENSITY_BINS}"
    print(f"  Bin {b+1}: {lo:.4f} – {hi:.4f} Gal/Mpc³  ({len(idx)} Quellen)")
    run_stack(idx, label)


# ──────────────────────────────────────────────
# 8. STACK 3: NACH REDSHIFT-BINS
# ──────────────────────────────────────────────

print(f"\n── Stack 3: {N_Z_BINS} Redshift-Bins ──")
z_bins = np.quantile(zred, np.linspace(0, 1, N_Z_BINS + 1))
print(f"  Redshift-Grenzen: {np.round(z_bins, 4)}")

for b in range(N_Z_BINS):
    lo, hi = z_bins[b], z_bins[b + 1]
    if b == N_Z_BINS - 1:
        mask = (zred >= lo) & (zred <= hi)
    else:
        mask = (zred >= lo) & (zred < hi)
    idx = np.where(mask)[0]
    label = f"zbin{b+1}_of{N_Z_BINS}"
    print(f"  Bin {b+1}: z = {lo:.4f} – {hi:.4f}  ({len(idx)} Quellen)")
    run_stack(idx, label)


print("\n✓ Fertig! Alle Stacks und der Dichte-Cube wurden als FITS-Dateien gespeichert.")