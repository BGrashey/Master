import os

import numpy as np
import matplotlib.pyplot as plt

from astropy.modeling import models, fitting
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d

from photutils.aperture import (
    CircularAperture,
    aperture_photometry,
    CircularAnnulus,
    ApertureStats,
)


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

        n_spec, n_y, n_x = cube.shape

        pad_xy  = 20
        pad_lam = 100

        x0 = max(x - pad_xy,  0);      x1 = min(x + pad_xy,  n_x)
        y0 = max(y - pad_xy,  0);      y1 = min(y + pad_xy,  n_y)
        s0 = max(self.center_slice - pad_lam, 0)
        s1 = min(self.center_slice + pad_lam, n_spec)

        self.x = x - x0
        self.y = y - y0
        self.center_idx = self.center_slice - s0

        self.wave_start = self.CRVAL3 + (s0 - self.CRPIX3) * self.CDELT3

        self.data = np.nan_to_num(cube[s0:s1, y0:y1, x0:x1], nan=0.0)
        if self.data.size == 0:
            raise ValueError(f"Object at ({self.ra}, {self.dec}) is fully outside cube bounds.")

        if self.data.shape != (2*pad_lam, 2*pad_xy, 2*pad_xy):
            import warnings
            warnings.warn(
                f"Sub-cube is truncated at the edge: shape={self.data.shape}. "
                "Results may be less reliable.",
                RuntimeWarning
            )

        self.catalog = catalog
        self.catalog_coord = catalog_skycoord

        self.wave, self.spec = self.get_spectrum()
        self.peak_flux, self.cont_fit, self.line_mask, self.center, self.popt = self.fit_model()
        self.flux_err, self.cont_err_raw = self.mc_flux_err()
        self.fwhm_kms = self.fwhm()
        self.snr_ = self.snr()
        self.g_band_mag, self.mag_err = self.get_g_band_mag()
        self.cont, self.cont_err = self.get_cont()
        self.ew_obs, self.ew, self.ew_err = self.ew()
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
        data_slice = np.nan_to_num(data_slice, nan=0.0, posinf=0.0, neginf=0.0)

        radii = np.arange(3, r_max + 1, 1)
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

        calibration = 1e-17

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

    def find_line_region(self, smooth_sigma=2, snr_threshold=2.0, search_halfwidth=20):
        smoothed = gaussian_filter1d(self.spec, sigma=smooth_sigma)

        search_window = (self.wave > self.lamda_center - search_halfwidth) & \
                        (self.wave < self.lamda_center + search_halfwidth)

        if not np.any(search_window):
            return np.nan, np.nan, np.nan

        noise_rms = np.nanstd(self.spec[~search_window])

        idx_in_window = np.where(search_window)[0]
        peak_idx = idx_in_window[np.argmax(smoothed[search_window])]
        peak_wave = self.wave[peak_idx]

        cont_level = np.nanmedian(self.spec[~search_window])
        threshold = cont_level + snr_threshold * noise_rms

        left = peak_idx
        while left > 0 and smoothed[left] > threshold:
            left -= 1
        right = peak_idx
        while right < len(smoothed) - 1 and smoothed[right] > threshold:
            right += 1

        return peak_wave, self.wave[left], self.wave[right]


    def fit_model(self):
        peak_wave, line_min, line_max = self.find_line_region()

        if np.isnan(peak_wave):
            return np.nan, np.nan, np.nan, np.nan, np.nan

        rough_width = max(line_max - line_min, 2.0)
        sigma_guess = rough_width / 4.0

        amp_guess = np.nanmax(self.spec) - np.nanmedian(self.spec)
        amp_max = max(3 * amp_guess, 1e-19)
        cont_guess = np.nanmedian(self.spec)

        p0 = [amp_guess, peak_wave, sigma_guess, cont_guess]

        bounds = (
            [0,          line_min - 3,  0.5,          -np.inf],
            [amp_max,    line_max + 3,  rough_width,   np.inf],
        )

        try:
            popt, _ = curve_fit(gaussian, self.wave, self.spec, p0=p0, bounds=bounds, maxfev=5000)
            _, mu, sigma, cont = popt

            fit_line_min = mu - 3 * sigma
            fit_line_max = mu + 3 * sigma
            line_mask = (self.wave > fit_line_min) & (self.wave < fit_line_max)

            flux = np.trapezoid(self.spec[line_mask] - cont, self.wave[line_mask])
        except RuntimeError:
            return np.nan, np.nan, np.nan, np.nan, np.nan

        return flux, cont, line_mask, mu, popt

    def mc_flux_err(self, n_iter=200):
        if np.isscalar(self.line_mask) and np.isnan(self.line_mask):
            return np.nan, np.nan

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

    def fwhm(self, r_spec=750):
        if np.isscalar(self.popt) and np.isnan(self.popt):
            return np.nan

        _, mu, sigma, _ = self.popt
        fwhm_obs = 2.3548 * sigma

        fwhm_inst_AA = mu / r_spec
        fwhm_intrinsic_AA = np.sqrt(max(fwhm_obs**2 - fwhm_inst_AA**2, 0))

        c_kms = 299792.458
        return fwhm_intrinsic_AA / mu * c_kms

    def snr(self):
        if np.isnan(self.peak_flux) or np.isnan(self.flux_err) or self.flux_err == 0:
            return np.nan
        return self.peak_flux / self.flux_err

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

        return ew_obs, ew, err, flux, flux_err, cont, cont_err, z, self.fwhm_kms, self.snr_

    def plot_ew(self, save_path=None, show=False):
        """
        save_path : str oder None — falls gesetzt, wird der Plot dort gespeichert
        show      : bool — ob der Plot interaktiv angezeigt werden soll (nur für Einzelfälle sinnvoll)
        """
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

        if save_path is not None:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()

        plt.close(fig)