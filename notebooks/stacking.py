from tools.stacking_revised import StackingFromSubcubes, load_subcubes_npz 

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

subcubes = load_subcubes_npz("/data/hetdex/u/bgrashey/data/subcubes_checked.npz")

stacker = StackingFromSubcubes(subcubes, kpc_pxl=2, npix=150)

stacked_cube = stacker.stack(
    do_sky_sub=False,
    do_cont_sub=True,
    normalize=False
)

stacked_nb = stacker.narrowband_from_cube(half_width=5, mode="sum")

import numpy as np

psf = stacker.stack_psf()

r_kpc, sb, sb_err = stacker.extract_sb_profile(
    stacked_nb,
    kpc_per_px=stacker.kpc_pxl,
    r_min=3,
    r_max=70,
    n_bins=20
)

center = np.unravel_index(np.nanargmax(psf), psf.shape)

r_psf, psf_points, psf_err = stacker.extract_sb_profile(
    psf,
    center=center,
    kpc_per_px=stacker.kpc_pxl,
    r_min=3,
    r_max=24,
    n_bins=25
)

sb, sb_err = sb*1e-17, sb_err*1e-17
norm = np.max(sb) / np.max(psf_points)
psf_scaled = psf_points * norm
psf_err = psf_err*norm

plt.errorbar(r_kpc, sb, yerr=sb_err)
plt.errorbar(r_psf, psf_scaled, yerr=psf_err)
plt.yscale("log")
plt.ylabel("SBe [erg/s/cm^2/arcsec^2]")
plt.xlabel("r [pkpc]")
#plt.ylim(1e-40, 1.2e-18)
plt.show()
plt.savefig("/data/hetdex/u/bgrashey/git/Master/plots/halo.pdf")
plt.close()