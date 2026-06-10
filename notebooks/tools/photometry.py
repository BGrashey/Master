import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry, ApertureStats

class Spectrum:
    
    def __init__(self, fits_cube_path, x0, y0, r_source):
        with fits.open(fits_cube_path) as cube:
            self.data = cube[0].data
            self.header = cube[0].header
            self.wcs = WCS(self.header)

        self.x0 = x0
        self.y0 = y0
        self.r_source = r_source
        self.r_in_sky = r_source * 2.
        self.r_out_sky = r_source * 3.
        
        self.aperture = CircularAperture((x0, y0), r=self.r_source)
        self.annulus = CircularAnnulus((x0, y0), r_in=self.r_in_sky,
                                       r_out=self.r_out_sky)

        self.wlgrid = self.grid()
        
        self.spec_final = None
        self.wcs = WCS(self.header)
        
        self.calibration_factor = 1e-17

    def grid(self):
        CRVAL = self.header['CRVAL3'] 
        CDELT = self.header['CDELT3'] 
        CRPIX = self.header['CRPIX3'] 
        N_WLS = self.data.shape[0] 

        indices = np.arange(N_WLS) 
        wlgrid = CRVAL + (indices - (CRPIX - 1)) * CDELT
        return wlgrid
    
    def measure_spec(self):
        calibration_factor = self.calibration_factor
        spec_flux_values = []
        area_ap = self.aperture.area
        
        for i in range(len(self.wlgrid)):
            image_slice = self.data[i, :, :]

            phot_table = aperture_photometry(image_slice, self.aperture)
            flux_ap = phot_table['aperture_sum'][0]

            annulus_stats = ApertureStats(image_slice, self.annulus)
            sky_median = annulus_stats.median
            
            sky_subtracted_flux = flux_ap - (sky_median * area_ap)
            calibrated_flux = sky_subtracted_flux * calibration_factor
            
            spec_flux_values.append(calibrated_flux)

        self.spec_final = np.array(spec_flux_values)
        return self.wlgrid, self.spec_final
    
    def measure_spec_unsubstracted(self):
        calibration_factor = self.calibration_factor
        spec_flux_values = []

        for i in range(len(self.wlgrid)):
            image_slice = self.data[i, :, :]
            phot_table = aperture_photometry(image_slice, self.aperture)
            flux_ap = phot_table['aperture_sum'][0]
            calibrated_flux = flux_ap * calibration_factor
            spec_flux_values.append(calibrated_flux)

        self.spec_final = np.array(spec_flux_values)
        return self.wlgrid, self.spec_final