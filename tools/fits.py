from astropy.io import fits
from astropy.io import WCS
from astropy.table import Table



# function to read Cubes
def read_cube(filename):
    """
    Reads the cube fits file and gives the data, header and WCS object
    
    :param filename: Absolute path to the cube
    """
    with fits.open(filename) as cube:
        data = cube[0].data
        header = cube[0].header
        wcs = WCS(header)
    
    return data, header, wcs


# function to read tables
def read_table(filename):
    """
    Reads a fits table and gives a pandas df
    
    :param filename: Absolute path to the table
    """
    tbl = Table.read(filename)
    df = tbl.to_pandas()

    return df