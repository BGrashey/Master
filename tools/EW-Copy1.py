import os

import numpy as np
import matplotlib.pyplot as plt

from tools.cubes import Cube

from astropy.modeling import models, fitting
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

from scipy.optimize import curve_fit

from photutils.aperture import (
    CircularAperture,
    aperture_photometry,
    CircularAnnulus,
    ApertureStats,
)

# define the gauss function for the fit later
def gaussian(x, amp, mu, sigma, cont):
    return cont + amp * np.exp(-0.5 * ((x - mu) / sigma)**2)


class Measurements:
    """
    Class to perform spectral and spatial measurements on 3D data cubes.
    """

    def __init__(
        self,
        cube,
        cube_header,
        coords: tuple,
        catalog=None,
        catalog_skycoord=None,
        degree: int = 2,
    ):
        """
        Initializes the Measurements class.

        Args:
            cube: Loaded cube data.
            cube_header: Header of the data cube.
            coords: Tuple containing (RA, DEC, redshift).
            catalog: Pandas DataFrame.
            catalog_skycoord: SkyCoord catalog.
            degree: Degree of the polynomial for continuum fit.
        """

        # preparing the slices for limiting the data
        self.CRVAL3 = cube_header["CRVAL3"]
        self.CRPIX3 = cube_header["CRPIX3"]
        self.CDELT3 = cube_header["CDELT3"]
        # get the coordinates
        self.ra, self.dec, self.z = coords
        wcs = WCS(cube_header, naxis=2)
        x, y = wcs.all_world2pix(self.ra, self.dec, 0)
        x, y = int(x), int(y)
        self.lamda_center = 1215.670 * (1 + self.z)
        self.center_slice = int((self.lamda_center - self.CRVAL3) / self.CDELT3 + self.CRPIX3)
        slice_lim = self.center_slice - 100, self.center_slice + 100
        self.wave_start = self.CRVAL3 + (slice_lim[0] - self.CRPIX3) * self.CDELT3
        x_lim = x - 20, x + 20
        y_lim = y - 20, y + 20
        self.x = x - x_lim[0]
        self.y = y - y_lim[0]
        self.center_idx = self.center_slice - slice_lim[0]
        # get the data
        self.data = cube[slice_lim[0]:slice_lim[1], y_lim[0]:y_lim[1], x_lim[0]:x_lim[1]]
        if self.data.size == 0:
            raise ValueError("Object outside cube bounds")
        # get the catalog
        self.catalog = catalog
        self.catalog_coord = catalog_skycoord
        # get the spectrum

        # self.start als erste Wellenlänge aus dem Slice berechnen
        self.wave, self.spec = self.get_spectrum()
        self.peak_flux, self.cont_fit, self.line_mask, self.center = self.fit_model()
        self.g_band_mag, self.mag_err = self.get_g_band_mag()
        self.cont = self.get_cont()
        self.redshift = self.center / 1215.670 - 1


    # ---------------------------------------------------------------------
    # Measurement functions
    # ---------------------------------------------------------------------
    def cog(self, r_max=15, threshold=0.05):
        """
        Perform a curve of growth to find the necessary aperture.
        Uses one wavelength slice at the estimated redshift.
        """

        data_slice = self.data[self.center_idx,:,:]

        radii = np.arange(1, r_max, 1)
        apertures = [
            CircularAperture((self.x, self.y), r=r) for r in radii
        ]

        annulus = CircularAnnulus((self.x, self.y), r_in=r_max+2, r_out=r_max+5)
        annulus_mask = annulus.to_mask(method="center")
        annulus_data = annulus_mask.multiply(data_slice)
        annulus_data = annulus_data[annulus_data != 0]
        if len(annulus_data) == 0:
            sky_median = 0.0
        else:
            sky_median = np.nanmedian(annulus_data)

        fluxes = []

        for ap in apertures:
            phot = aperture_photometry(data_slice, ap)
            aperture_flux = phot["aperture_sum"][0]

            aperture_area = ap.area
            sky_flux = sky_median * aperture_area

            fluxes.append(aperture_flux - sky_flux)

        fluxes_arr = np.array(fluxes)

        fluxes_norm = fluxes_arr / fluxes_arr[-1]
        flux_grad = np.diff(fluxes_norm)
        conv = np.where(flux_grad < threshold)[0]
        r_opt = radii[conv[0] + 1] if len(conv) else r_max

        return r_opt
    
    def get_spectrum(self):
        r = self.cog()
        r_in = r + 2
        r_out = r + 5

        aperture = CircularAperture((self.x, self.y), r=r)
        annulus = CircularAnnulus((self.x, self.y), r_in=r_in, r_out=r_out)
        aperture_area = aperture.area

        N_wls = self.data.shape[0]
        indices = np.arange(N_wls)
        wl_grid = self.wave_start + indices * self.CDELT3

        calibration = 1e-17 # erg / s / cm**2 / AA

        spec_flux_values = []

        for i in range(len(wl_grid)):
            image_slice = self.data[i,:,:]
            image_slice = np.nan_to_num(image_slice, nan=0.0, posinf=0.0, neginf=0.0)

            phot = aperture_photometry(image_slice, aperture)
            flux = phot["aperture_sum"][0]
            annulus_mask = annulus.to_mask(method="center")
            annulus_data = annulus_mask.multiply(image_slice)
            annulus_data = annulus_data[annulus_data != 0]
            if len(annulus_data) == 0:
                sky_median = 0.0
            else:
                sky_median = np.nanmedian(annulus_data)

            if not np.isfinite(sky_median):
                sky_median = 0.
        
            substracted_flux = flux - (sky_median * aperture_area)
            calibrated = substracted_flux * calibration

            spec_flux_values.append(calibrated)
        
        spec_final = np.array(spec_flux_values)

        return wl_grid, spec_final
    
    def fit_model(self):
        center_guess = self.lamda_center
        center_idx = np.argmin(np.abs(self.wave - self.lamda_center))
        amp_guess = self.spec[center_idx] - np.nanmedian(self.spec)
        p0 = amp_guess, center_guess, 2, np.nanmedian(self.spec)
        
        bounds = (
                [0,          self.lamda_center - 20,  0.5,  -np.inf],
                [np.inf,     self.lamda_center + 20,  15,    np.inf],
                )

        try:
            popt, _ = curve_fit(gaussian, self.wave, self.spec, p0, bounds=bounds, maxfev=5000)

            _, mu, sigma, cont = popt

            line_min = mu - 3 * sigma #2.355
            line_max = mu + 3 * sigma

            line_mask = (self.wave > line_min) & (self.wave < line_max)

            flux = np.trapezoid(
                (self.spec[line_mask] - cont),
                self.wave[line_mask]
            )
        except RuntimeError:
            return np.nan, np.nan, np.nan, np.nan

        return flux, cont, line_mask, mu
    
    def get_g_band_mag(self, tol=2.):
        if isinstance(self.ra, str):
            c_obj = SkyCoord(self.ra, self.dec, frame="icrs")
        else:
            c_obj = SkyCoord(self.ra, self.dec, frame="icrs", unit=u.deg)
        
        idx, d2d, _ = c_obj.match_to_catalog_sky(self.catalog_coord)

        if d2d.to(u.arcsec).value < tol:
            return self.catalog.iloc[idx]["g_cmodel_mag"], self.catalog.iloc[idx]["g_cmodel_magerr"]
        else:
            return np.nan, np.nan

    def cont_hsc(self):
        g_mag = self.g_band_mag
        c, lam_eff, band_width = 2.99792458e18, 4726, 1468
        corr = (self.center / lam_eff) ** (-2)
        f_lambda = 10**(-0.4 * (g_mag + 48.6)) * c / lam_eff**2
        f_cont = (f_lambda - self.peak_flux / band_width) * corr

        return f_cont

    def get_cont(self):
        if np.isnan(self.g_band_mag):
            cont= self.cont_fit
        else:
            cont = self.cont_hsc()
        
        if cont < 0:
            return np.nanstd(self.spec[~self.line_mask])
        else:
            return cont
        
    def ew(self):
        ew_obs = self.peak_flux / self.cont
        ew = ew_obs / (1 + self.z)

        return ew_obs, ew
    
    def measure_ew(self):
        ew_obs, ew = self.ew()
        flux = self.peak_flux
        cont = self.cont
        z = self.redshift

        return ew_obs, ew, flux, cont, z
