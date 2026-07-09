%load_ext autoreload
%autoreload 2

from astropy.table import Table

# catalog:
table = "/data/hetdex/u/bgrashey/data_/fof_run_3_5_cnnscored.fits"

tbl = Table.read(table)
Catalog = tbl.to_pandas()

#Catalog


from astropy.table import Table
import os
from tools.cubes import Cube
import numpy as np
from tools.EQW import Measurements
from astropy.io import fits
from astropy.coordinates import SkyCoord

directory = "/data/hetdex/u/mxhf/VDFI/VirusDeep/stackmf/ssa22_fullfp/"
#directory = "/data/hetdex/u/jsnigula/VirusDeep/stack/ssa22_fullfp"
file_stack = "ssa22_fullfp_stack.fits"
#file_stack = "ssa22_fullfp_stack_20260115.fits"

stack = os.path.join(directory, file_stack)
stack_cube = Cube(stack)
hsc_file="/data/hetdex/u/mxhf/VDFI/joint/ssa22_hsc_pdr3_wide.fits"
#with fits.open(hsc_file) as hdul:
#            df = Table(hdul[1].data).to_pandas()


with fits.open(hsc_file) as hdul:
    hsc_df = Table(hdul[1].data).to_pandas()

hsc_coords = SkyCoord(ra=hsc_df["ra"].values, 
                      dec=hsc_df["dec"].values, 
                      unit="deg", frame="icrs")        
        
        
        
table = Table(names=("ID", "EQ_WIDTH", "EQ_ERROR", "EQ_WIDTH_HSC", "EQ_HSC_ERROR", "FLUX", "ERROR", "REDSHIFT", "CONT", "CONT_HSC"),
             dtype=("i4", "f8", "f8", "f8", "f8", "f8", "f8", "f4", "f8", "f8"))

out_dir = "/data/hetdex/u/bgrashey/data/eq_plots" 


N = len(Catalog["ra"])
#N = 10

"""bad_objects = [
    11, 13, 15, 16, 17, 18, 19, 21, 
    23, 24, 25, 26, 27, 29, 31, 33, 
    41, 42, 43, 45, 53, 56, 61, 62, 64, 
    68, 69, 70, 71, 78, 92, 100, 106, 
    108, 110, 111, 112, 113, 115, 116, 
    117, 118, 119, 120, 121, 122, 123, 
    125, 126, 129, 130, 131, 134, 135, 
    137, 138, 139, 140, 142, 143, 144, 
    145, 146, 147, 154
]"""
bad_objects = []

for n in range(N):
    
    if n in bad_objects:
        
        print(f"Bad object {n+1}")
        continue
    
    print(f"\nInitialising Object {n+1}/{N}")
    
    try:
        coords = (
            Catalog["ra"][n],
            Catalog["dec"][n],
            Catalog['z_vdfi'][n]
        )
        x = 1 #Catalog["xsigma_2"][n]
        y = 1# Catalog["ysigma_2"][n]
        size = (
            x,
            y
        )
        z_s = 1 # Catalog["zextent_2"][n]
        
        meas = Measurements(stack_cube, coords, size, hsc_df, hsc_coords, degree=1)
        
        EW, ER, EW_HSC, ERR, F, E, z, C, C_ = meas.measure_ew()
        
        
        table.add_row([n, EW, ER, EW_HSC, ERR, F, E, z, C, C_])
        
        #meas.plot_ew()
        
        print("done")
    
    except Exception as e:
        print(f"Object {n} failed: {e}")
        
        table.add_row([n, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
        
        continue


out_dir = "/data/hetdex/u/bgrashey/data/"        
name = "Catalog_EW_new_unfiltered.fits"
out_path = os.path.join(out_dir, name)
table.write(out_path, overwrite=True)

print("writing successfull")
