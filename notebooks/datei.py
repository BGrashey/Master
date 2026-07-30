import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ra",   default="RA",  help="Spaltenname RA")
parser.add_argument("--dec",  default="DEC", help="Spaltenname DEC")
parser.add_argument("--z",    default="z",   help="Spaltenname Redshift")
parser.add_argument("--catalog",  required=True, help="Pfad zum Eingabe-Katalog")
args = parser.parse_args()

COL_RA  = args.ra
COL_DEC = args.dec
COL_z   = args.z
table_path = args.catalog

from astropy.table import Table
import os
import numpy as np
from tools.EW import Measurements
from astropy.io import fits
from astropy.coordinates import SkyCoord

# catalog:
tbl = Table.read(table_path)
Catalog = tbl.to_pandas()
Catalog = Catalog.reset_index(drop=True)

#directory = "/data/hetdex/u/mxhf/VDFI/VirusDeep/stackmf/ssa22_fullfp/"
#file_stack = "ssa22_fullfp_stack.fits"
#stack = os.path.join(directory, file_stack)
stack = "/data/hetdex/u/bgrashey/cubes/ssa22_fullfp_stack.fits"
cube = fits.open(stack, memmap=True)
data = cube[0].section
header = cube[0].header

hsc_file = "/data/hetdex/u/mxhf/VDFI/joint/ssa22_hsc_pdr3_wide.fits"
with fits.open(hsc_file) as hdul:
    hsc_df = Table(hdul[1].data).to_pandas()
hsc_coords = SkyCoord(ra=hsc_df["ra"].values, 
                      dec=hsc_df["dec"].values, 
                      unit="deg", frame="icrs")        

# Neue Spalten als Listen initialisieren
new_cols = {
    "REDSHIFT": [], "EW": [], "EW_OBS": [], "EW_ERR": [],
    "FLUX": [], "FLUX_ERR": [], "CONT": [], "CONT_ERR": [],
    "FWHM": [], "SNR": []
}

N = len(Catalog[COL_RA])
for n in range(N):
    print(f"\nInitialising Object {n+1}/{N}")
    
    try:
        coords = (
            Catalog[COL_RA][n],
            Catalog[COL_DEC][n],
            Catalog[COL_z][n]
        )
        
        meas = Measurements(data, header, coords, hsc_df, hsc_coords)
        EW_OBS, EW, EW_ERR, F, F_ERR, C, C_ERR, z, fwhm, snr = meas.measure_ew()
        
        new_cols["REDSHIFT"].append(z)
        new_cols["EW"].append(EW)
        new_cols["EW_OBS"].append(EW_OBS)
        new_cols["EW_ERR"].append(EW_ERR)
        new_cols["FLUX"].append(F)
        new_cols["FLUX_ERR"].append(F_ERR)
        new_cols["CONT"].append(C)
        new_cols["CONT_ERR"].append(C_ERR)
        new_cols["FWHM"].append(fwhm)
        new_cols["SNR"].append(snr)
        
        print("done")
    
    except Exception as e:
        print(f"Object {n} failed: {e}")
        for key in new_cols:
            new_cols[key].append(np.nan)

# Neue Spalten an den originalen Katalog hängen
for col_name, values in new_cols.items():
    tbl[col_name] = np.array(values, dtype="f8")

# Klassifikations-Wahrscheinlichkeit berechnen
from line_classifier.probs.classification_prob import source_prob
import configparser

config_file = "/data/hetdex/u/bgrashey/notebooks/line_prob/hetdex_line_classification/tests/data/universe.cfg"
flim_file = "/data/hetdex/u/bgrashey/notebooks/Line_flux_limit_5_sigma_baseline.dat"

config = configparser.ConfigParser()
config.read(config_file)

z   = tbl["REDSHIFT"]
F   = tbl["FLUX"]
E   = tbl["FLUX_ERR"]
eq  = tbl["EW_OBS"]
eq_E = tbl["EW_ERR"]

p = source_prob(config, [1]*len(z), [1]*len(z), z, F, E, eq, eq_E,
                None, None, None, None, None, flim_file, ignore_noise=True)
tbl["PROB"] = p

out_path = table_path
tbl.write(out_path, overwrite=True)
print(f"Catalog saved at {out_path}")

cube.close()
