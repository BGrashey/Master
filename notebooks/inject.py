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

bins = [(0.05, 0.1), (0.1, 0.5), (0.5, 1), (1, 10), (10, 100), (100,1000), (1000, 2000), (2000, 10000)]

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
    outpath="/data/hetdex/u/bgrashey/cubes/test.fits",
    header_params=header
)

df_output = pd.DataFrame(new_catalog)
tbl = Table.from_pandas(df_output)
tbl.write("/data/hetdex/u/bgrashey/data_/injected_sources.fits", overwrite=True)

from tools.cubes import fits_ifu_to_zarr

fits_ifu_to_zarr("/data/hetdex/u/bgrashey/cubes/test.fits", "/data/hetdex/u/bgrashey/cubes/test.zarr")