import os

import numpy as np
import matplotlib.pyplot as plt

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

        self.CRVAL3 = cube_header["CRVAL3"]
        self.CRPIX3 = cube_header["CRPIX3"]
        self.CDELT3 = cube_header["CDELT3"]

        self.ra, self.dec, self.z = coords
        wcs = WCS(cube_header, naxis=2)
        x, y = wcs.all_world2pix(self.ra, self.dec, 0)
        x, y = int(x), int(y)

        self.lamda_center = 1215.670 * (1 + self.z)
        self.center_slice = int((self.lamda_center - self.CRVAL3) / self.CDELT3 + self.CRPIX3)

        # --- cube shape ---
        n_spec, n_y, n_x = cube.shape

        # --- clamp all slice limits to valid cube bounds ---
        pad_xy  = 20
        pad_lam = 100

        x0 = max(x - pad_xy,  0);      x1 = min(x + pad_xy,  n_x)
        y0 = max(y - pad_xy,  0);      y1 = min(y + pad_xy,  n_y)
        s0 = max(self.center_slice - pad_lam, 0)
        s1 = min(self.center_slice + pad_lam, n_spec)

        # position of the target *inside* the sub-cube
        self.x = x - x0
        self.y = y - y0
        self.center_idx = self.center_slice - s0

        # wavelength at the first kept spectral channel
        self.wave_start = self.CRVAL3 + (s0 - self.CRPIX3) * self.CDELT3

        self.data = np.nan_to_num(cube[s0:s1, y0:y1, x0:x1], nan=0.0)
        if self.data.size == 0:
            raise ValueError(f"Object at ({self.ra}, {self.dec}) is fully outside cube bounds.")

        # warn if the sub-cube is smaller than intended (edge case)
        if self.data.shape != (2*pad_lam, 2*pad_xy, 2*pad_xy):
            import warnings
            warnings.warn(
                f"Sub-cube is truncated at the edge: shape={self.data.shape}. "
                "Results may be less reliable.",
                RuntimeWarning
            )
        # get the catalog
        self.catalog = catalog
        self.catalog_coord = catalog_skycoord
        # get the spectrum

        # self.start als erste Wellenlänge aus dem Slice berechnen
        self.wave, self.spec = self.get_spectrum()
        self.peak_flux, self.cont_fit, self.line_mask, self.center, self.popt = self.fit_model()
        self.flux_err, self.cont_err_raw = self.mc_flux_err()
        self.g_band_mag, self.mag_err = self.get_g_band_mag()
        self.cont, self.cont_err = self.get_cont()
        self.ew_obs, self.ew, self.ew_err = self.ew()
        self.redshift = self.center / 1215.670 - 1


    # ---------------------------------------------------------------------
    # Measurement functions
    # ---------------------------------------------------------------------
    def cog(self, r_max=12, threshold=0.05):
        """
        Perform a curve of growth to find the necessary aperture.
        Uses one wavelength slice at the estimated redshift.
        """

        data_slice = self.data[self.center_idx,:,:]
        data_slice = np.nan_to_num(data_slice, nan=0.0, posinf=0.0, neginf=0.0)

        radii = np.arange(4, r_max, 1)
        apertures = [
            CircularAperture((self.x, self.y), r=r) for r in radii
        ]

        annulus = CircularAnnulus((self.x, self.y), r_in=r_max+2, r_out=r_max+5)
        annulus_mask = annulus.to_mask(method="center")
        annulus_data = annulus_mask.multiply(data_slice)
        annulus_data = annulus_data[annulus_data != 0]
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

        return max(4, r_opt)
    
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
        p0 = np.nanmax(self.spec), center_guess, 2, np.nanmedian(self.spec)

        try:
            popt, _ = curve_fit(gaussian, self.wave, self.spec, p0)

            _, mu, sigma, cont = popt

            line_min = mu - 3 * sigma #2.355
            line_max = mu + 3 * sigma

            line_mask = (self.wave > line_min) & (self.wave < line_max)

            flux = np.trapezoid(
                (self.spec[line_mask] - cont),
                self.wave[line_mask]
            )
        except RuntimeError:
            return np.nan, np.nan, np.nan, np.nan, np.nan

        return flux, cont, line_mask, mu, popt
    
    def mc_flux_err(self, n_iter=1000):
        noise_rms = np.nanstd(self.spec[~self.line_mask])
        center_idx = np.argmin(np.abs(self.wave - self.lamda_center))
        fluxes = []
        conts = []

        for _ in range(n_iter):
            perturbed = self.spec + np.random.normal(0, noise_rms, size=self.spec.shape)
            try:
                amp_guess = perturbed[center_idx] - np.nanmedian(perturbed)
                p0 = amp_guess, self.lamda_center, 2, np.nanmedian(perturbed)
                bounds = (
                    [-np.inf, self.lamda_center - 20, 0.5, -np.inf],
                    [ np.inf, self.lamda_center + 20, 15,   np.inf],
                          )
                popt, _ = curve_fit(gaussian, self.wave, perturbed, p0, bounds=bounds, maxfev=5000)
                amp, _, sigma, cont = popt
                fluxes.append(amp * abs(sigma) * np.sqrt(2 * np.pi))
                conts.append(cont)
            except RuntimeError:
                continue

        fluxes_arr = np.array(fluxes)
        conts_arr = np.array(conts)
        return np.std(fluxes_arr), np.std(conts_arr)
    
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

        cont_err = np.sqrt(
            (self.flux_err / band_width)**2 +
            (f_lambda * np.log(10) * 0.4 * self.mag_err)**2
        )

        return f_cont, cont_err

    def get_cont(self):
        if np.isnan(self.g_band_mag):
            cont, err = self.cont_fit, self.cont_err_raw
        else:
            cont, err = self.cont_hsc()
        
        if np.isnan(cont) or cont <= 0:
            if self.line_mask is not np.nan:
                noise = np.nanstd(self.spec[~self.line_mask])
                return noise, noise
            else:
                return np.nan, np.nan
        return cont, err 
        
    def ew(self):
        if np.isnan(self.peak_flux) or np.isnan(self.cont) or self.cont == 0:
            return np.nan, np.nan, np.nan

        ew_obs = self.peak_flux / self.cont
        ew = ew_obs / (1 + self.z)

        try:
            rel_err_sq = (self.flux_err / self.peak_flux)**2 + (self.cont_err / self.cont)**2
            err = ew_obs * np.sqrt(rel_err_sq)
        except:
            err = np.nan

        return ew_obs, ew, err
    
    def measure_ew(self):
        ew_obs, ew, err = self.ew_obs, self.ew, self.ew_err
        flux = self.peak_flux
        cont = self.cont
        z = self.redshift
        flux_err = self.flux_err
        cont_err = self.cont_err

        return ew_obs, ew, err, flux, flux_err, cont, cont_err, z
    
    def plot_ew(self):
        ew = self.ew
        spec = self.spec
        wave = self.wave
        cont = self.cont
        amp, mu, sig, con = self.popt
        gauss = gaussian(wave, amp, mu, sig, con)
        line_mask = self.line_mask
        
        fig, ax = plt.subplots(figsize=(7,5))
        
        ax.plot(wave, spec, color="blue", lw=1, label="Flux")
        ax.plot(wave, gauss, color="red", lw=2, ls=":", alpha=0.5, label="Gauss Fit")
        ax.fill_between(wave[line_mask], spec[line_mask], cont, color="grey", alpha=0.3, label="Line Region")
        
        ax.axhline(y=cont, color="green", ls="--", lw=1, label="Cont Level")
        ax.set_xlabel("Wavelength [Å]")
        ax.set_ylabel(r"Flux $\frac{erg}{s \, cm^2 \, \AA}$")
        ax.legend(loc="best")
        ax.set_title(f"EW = {ew:.1f} [Å], z = {self.redshift:.1f}")

        plt.tight_layout()
        plt.show()
        plt.close(fig)
        
        
