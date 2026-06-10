import numpy as np

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.stats import SigmaClip

from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry, ApertureStats


class Cube:

    def __init__(self, filename):
        self.file = filename
        with fits.open(filename) as cube:
            self.data = cube[0].data
            self.header = cube[0].header
            self.wcs = WCS(self.header)

    def get_data(self):
        return self.data, self.header, self.wcs
    
    def world2pix(self, Ra, Dec):
        
        if isinstance(Ra, str):
            coord = SkyCoord(Ra, Dec, frame="icrs")
        else:
            coord = SkyCoord(Ra, Dec, frame="icrs", unit="deg")
                
        ra_deg = coord.ra
        dec_deg = coord.dec

        wcs_2d = self.wcs.celestial
        x0, y0 = wcs_2d.all_world2pix(ra_deg, dec_deg, 0)
    
        return x0, y0
    
    def pix2world(self, x, y):
        wcs_2d = self.wcs.celestial
        ra_deg, dec_deg = wcs_2d.all_pix2world(x, y, 0)
        coord = SkyCoord(ra_deg, dec_deg, frame="icrs", unit="deg")
        Ra = coord.ra.to_string(unit='hour', sep='hms', precision=2)
        Dec = coord.dec.to_string(unit='degree', sep='dms', precision=1)
    
        return Ra, Dec
    
    def parse_coords(self, ra, dec):
        
        if isinstance(ra, str) or isinstance(dec, str):
            c = SkyCoord(ra, dec, frame="icrs")
        else:
            c = SkyCoord(ra, dec, frame="icrs", unit="deg")
        
        return c.ra.deg, c.dec.deg
    
    def get_spectrum(self, x0, y0, r_source):

        # get the sky radii
        if r_source == 2:
            r_in_sky = r_source + 10
            r_out_sky = r_source + 13
        else:
            r_in_sky = 12
            r_out_sky = 15
        
        # get the aperture and annulus
        aperture = CircularAperture((x0,y0), r=r_source)
        annulus = CircularAnnulus((x0,y0), r_in=r_in_sky, r_out=r_out_sky)

        # get the wavelength grid
        CRVAL = self.header['CRVAL3'] 
        CDELT = self.header['CDELT3'] 
        CRPIX = self.header['CRPIX3'] 
        N_WLS = self.data.shape[0] 

        indices = np.arange(N_WLS) 
        wlgrid = CRVAL + (indices - (CRPIX - 1)) * CDELT

        #calibration factor of the detector
        calibration_factor = 1e-17 # erg / s / cm**2 / AA
        
        # sigma clipping
        sigclip = SigmaClip(sigma=3., maxiters=5)

        # do the spectrum measurement
        spec_flux_values = []
        area_ap = aperture.area

        for i in range (len(wlgrid)):
            image_slice = self.data[i, :, :]

            phot_table = aperture_photometry(image_slice, aperture)
            flux_ap = phot_table['aperture_sum'][0]

            annulus_stats = ApertureStats(
                image_slice,
                annulus,
                sigma_clip=sigclip
                
            )
            sky_median = annulus_stats.mean
            
            if not np.isfinite(sky_median):
                sky_median = 0.
            
            sky_subtracted_flux = flux_ap  - (sky_median * area_ap)
            calibrated_flux = sky_subtracted_flux * calibration_factor
            
            spec_flux_values.append(calibrated_flux)
        
        spec_final = np.array(spec_flux_values)
        
        return wlgrid, spec_final
    
    
#--------------------------------------------------------------
# function that checks the mag of a source to validate it as a source via hsc
#--------------------------------------------------------------

def mag_AB(
    flux,
    lambda_eff: float = 4726,
    band_width: float = 1468
    ):
    """Calculates the AB magnitude for a given filter and line_flux"""
    
    c = 2.99792458e18  # Å/s
    f_nu = flux / band_width * lambda_eff**2 / c
    
    m_AB = -2.5 * np.log10(f_nu) - 48.60
    
    return m_AB

import zarr
import numpy as np
from astropy.io import fits

def fits_ifu_to_zarr(fits_path, zarr_path, chunks=(100, 128, 128)):
    root = zarr.open_group(zarr_path, mode="w")

    with fits.open(fits_path) as hdul:
        for i, hdu in enumerate(hdul):
            if hdu.data is None:
                continue
            name = hdu.name or f"ext_{i}"
            
            # Chunks nur für 3D-Würfel anpassen, sonst auto
            c = chunks if hdu.data.ndim == 3 else "auto"
            
            arr = root.create_array(
                name,
                data=hdu.data,
                chunks=c,
                compressors=None,
            )
            arr.attrs["fits_header"] = dict(hdu.header)
            print(f"  {name}: {hdu.data.shape}, chunks={arr.chunks}")

    print(zarr.open(zarr_path).tree())