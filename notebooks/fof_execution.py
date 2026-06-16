import zarr
import dask
import dask.array as da
from astropy.wcs import WCS
from astropy.io import fits

from tools.fof_min import fof_minimal_zarr, catalog_to_wcs_table

sn_cube_path = "/data/hetdex/u/bgrashey/cubes/test.zarr"
fits_header_path = "/data/hetdex/u/bgrashey/cubes/ssa22_fullfp_stack.fits"
output_catalog_path = "/data/hetdex/u/bgrashey/cubes/gefundene_quellen.fits"
sn_threshold = 0.5
linking_length = 1.5


sn_cube = da.from_zarr(sn_cube_path, component="PRIMARY")

binary_mask = sn_cube > sn_threshold

catalog_delayed = fof_minimal_zarr(binary_mask, linking_length=linking_length)

cat_computed = dask.compute(catalog_delayed)[0]

header = fits.getheader(fits_header_path)

final_table = catalog_to_wcs_table(cat_computed, wcs_header=header)

final_table.write(output_catalog_path, overwrite=True)

print(f"Fertig: {len(final_table)} Quellen und '{output_catalog_path}' gespeichert.")