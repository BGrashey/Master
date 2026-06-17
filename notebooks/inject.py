from tools.simulation import inject_sources_into_cube, save_cube_fits, fits_header, build_fov_mask, generate_positions
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import pandas as pd

flux_bins = np.linspace(1000, 9000, 11)

region = "/data/hetdex/u/bgrashey/regions/fov.reg"

with fits.open("/data/hetdex/u/bgrashey/cubes/ssa22_fullfp_stack.fits") as hdul:
    wcs = WCS(hdul[0].header)
    cube = hdul[0].data.astype(np.float32)
    header = hdul[0].header

nz, ny, nx = cube.shape
    
from astropy.table import Table

fits_table = Table.read("/data/hetdex/u/bgrashey/data_/combined_manual_vdfi_matched.fits")

real_catalog_world = []
for row in fits_table:
    real_catalog_world.append([
        float(row['ra_vdfi']), 
        float(row['dec_vdfi']), 
        float(row['z_vdfi'])
    ])
    
fov = build_fov_mask(
        nz=nz, ny=ny, nx=nx, 
        wcs=wcs,
        reg_file=region
    )

chunk_size = 10
chunks = [(i, min(i + chunk_size, nz)) for i in range(0, nz, chunk_size)]

fov_mask = build_fov_mask(nz, ny, nx, wcs, reg_file=region)

bins = [
    (0.5,  1.0),    # ~29-28 mag — unter Detektionslimit
    (1.0,  2.0),    # ~28-27 mag — Detektionslimit
    (2.0,  5.0),    # ~27-26 mag — schwach
    (5.0,  10.0),   # ~26-25 mag — mittel
    (10.0, 50.0),   # ~25-24 mag — hell
    (50.0, 200.0),  # ~24-23 mag — sehr hell
]

positions = generate_positions(
    nz, ny, nx, wcs,
    existing_catalog=real_catalog_world,
    chunks=chunks,
    n_bins=len(bins),
    n_per_bin=10,
    fov_mask=fov_mask,
 )

flux_cube, new_catalog = inject_sources_into_cube(
    cube, positions, bins,
)

with fits.open("/data/hetdex/u/bgrashey/cubes/ssa22_fullfp_stack.fits") as hdul_err:
    error_cube = hdul_err[1].data.astype(np.float32)

mask = ~(error_cube > 0)

with np.errstate(divide='ignore', invalid='ignore'):
    flux_cube /= error_cube

flux_cube[mask] = np.nan
del error_cube, mask

save_cube_fits(
    cube=flux_cube,
    outpath="/data/hetdex/u/bgrashey/cubes/test_.fits",
    header_params=header
)

df_output = pd.DataFrame(new_catalog)
tbl = Table.from_pandas(df_output)
tbl.write("/data/hetdex/u/bgrashey/data_/injected_sources_real.fits", overwrite=True)

from tools.cubes import fits_ifu_to_zarr

fits_ifu_to_zarr("/data/hetdex/u/bgrashey/cubes/test_.fits", "/data/hetdex/u/bgrashey/cubes/injected.zarr")