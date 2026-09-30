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

stacked_nb = stacker.narrowband_from_cube(half_width=5, mode="mean")

r_kpc, sb, sb_err = stacker.extract_sb_profile(
    stacked_nb,
    kpc_per_px=stacker.kpc_pxl, 
    r_max=70, 
    n_bins=15
)

plt.plot(r_kpc, sb*1e-17+1e-20)
plt.yscale("log")
plt.ylabel("SBe [erg/s/cm^2/arcsec^2]")
plt.xlabel("r [pkpc]")
#plt.ylim(1e-40, 1.2e-18)
plt.savefig("/data/hetdex/u/bgrashey/halo.pdf")
plt.close()