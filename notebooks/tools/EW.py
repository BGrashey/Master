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
from scipy.signal import find_peaks

from photutils.aperture import (
    CircularAperture,
    aperture_photometry,
    CircularAnnulus,
    ApertureStats,
)


def gaussian(x, amp, mu, sigma, cont):
    return cont + amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _is_nan_scalar(value):
    return np.isscalar(value) and np.isnan(value)


class Measurements:
    """
    Class to perform spectral and spatial measurements on 3D data cubes.

    Args:
        cube: Loaded cube data.
        cube_header: Header of the data cube.
        coords: Tuple containing (RA, DEC, redshift).
        catalog: Pandas DataFrame.
        catalog_skycoord: SkyCoord catalog.
        degree: Degree of the polynomial for continuum fit.
        detect_snr_threshold: Mindest-SNR (relativ zum Kontinuum-Rausch-RMS),
            damit ein Peak ueberhaupt als Linienkandidat gilt.
        min_amp_snr: Mindest-SNR der gefitteten Gauss-Amplitude
            (amp / amp_err aus der Fit-Kovarianzmatrix), damit ein
            konvergierter Fit als signifikant gilt.
        min_mc_success_rate: Mindestanteil erfolgreicher MC-Iterationen,
            damit flux_err/cont_err_raw als verlaesslich gelten.
    """

    def __init__(
        self,
        cube,
        cube_header,
        error_cube,
        coords: tuple,
        catalog=None,
        catalog_skycoord=None,
        degree: int = 2,
        detect_snr_threshold: float = 2.0,
        min_amp_snr: float = 2.0,
        min_mc_success_rate: float = 0.5,
    ):

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

        pad_xy = 20
        pad_lam = 100

        x0 = max(x - pad_xy, 0)
        x1 = min(x + pad_xy, n_x)
        y0 = max(y - pad_xy, 0)
        y1 = min(y + pad_xy, n_y)
        s0 = max(self.center_slice - pad_lam, 0)
        s1 = min(self.center_slice + pad_lam, n_spec)

        self.x = x - x0
        self.y = y - y0
        self.center_idx = self.center_slice - s0

        self.wave_start = self.CRVAL3 + (s0 - self.CRPIX3) * self.CDELT3

        self.data = np.nan_to_num(cube[s0:s1, y0:y1, x0:x1], nan=0.0)
        self.error = np.nan_to_num(error_cube[s0:s1, y0:y1, x0:x1], nan=0.0)
        if self.data.size == 0:
            raise ValueError(f"Object at ({self.ra}, {self.dec}) is fully outside cube bounds.")

        if self.data.shape != (2 * pad_lam, 2 * pad_xy, 2 * pad_xy):
            import warnings
            warnings.warn(
                f"Sub-cube is at the edge: shape={self.data.shape}. "
                "Results may be less reliable.",
                RuntimeWarning
            )

        self.catalog = catalog
        self.catalog_coord = catalog_skycoord

        self.detect_snr_threshold = detect_snr_threshold
        self.min_amp_snr = min_amp_snr
        self.min_mc_success_rate = min_mc_success_rate

        self.wave, self.spec, self.error = self.get_spectrum()

        fit_result = self.fit_model()
        self.peak_flux = fit_result["flux"]
        self.flux_trapz = fit_result["flux_trapz"]
        self.cont_fit = fit_result["cont"]
        self.line_mask = fit_result["line_mask"]
        self.center = fit_result["center"]
        self.popt = fit_result["popt"]
        self.amp_err = fit_result["amp_err"]
        self.chi2_red = fit_result["chi2_red"]
        self.fit_significant = fit_result["fit_significant"]
        self.fwhm_kms_err = fit_result["fwhm_kms_err_"]
        self.z_err = fit_result["z_err_"]


        self.flux_err, self.cont_err_raw, self.center_err, self.width_err, self.mc_success_rate = self.mc_flux_err()
        self.fwhm_kms, _ = self.fwhm()
        self.snr_ = self.snr()
        self.g_band_mag, self.mag_err = self.get_g_band_mag()
        self.cont, self.cont_err, self.cont_is_noise_floor = self.get_cont()
        self.ew_obs, self.ew, self.ew_err = self.ew()
        self.redshift, self.redshift_err = (self.center / 1215.670 - 1 if not _is_nan_scalar(self.center) else np.nan), (self.center_err / 1215.670 if not _is_nan_scalar(self.center_err) else np.nan)

        self.detected = bool(self.fit_significant) and not _is_nan_scalar(self.peak_flux)

    # ---------------------------------------------------------------------
    # Measurement functions
    # ---------------------------------------------------------------------
    def cog(self, r_max=12, threshold=0.05, n_avg_channels=5):
        """
        Perform a curve of growth to find the necessary aperture.
        """
        half = n_avg_channels // 2
        k0 = max(self.center_idx - half, 0)
        k1 = min(self.center_idx + half + 1, self.data.shape[0])

        data_slice = np.nanmean(self.data[k0:k1, :, :], axis=0)
        data_slice = np.nan_to_num(data_slice, nan=0.0, posinf=0.0, neginf=0.0)

        radii = np.arange(3, r_max + 1, 1)
        apertures = [
            CircularAperture((self.x, self.y), r=r) for r in radii
        ]

        annulus = CircularAnnulus((self.x, self.y), r_in=r_max + 2, r_out=r_max + 5)
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
        err_values = []
        
        has_error_cube = (self.error is not None) and (np.any(self.error > 0))

        for i in range(len(wl_grid)):
            image_slice = self.data[i, :, :]
            image_slice = np.nan_to_num(image_slice, nan=0.0, posinf=0.0, neginf=0.0)
            
            if has_error_cube:
                error_slice = self.error[i, :, :]
                error_slice = np.nan_to_num(error_slice, nan=1.0, posinf=1.0, neginf=1.0)

                phot = aperture_photometry(image_slice, aperture, error=error_slice)
                flux = phot["aperture_sum"][0]
                err = phot["aperture_sum_err"][0]
            else:
                phot = aperture_photometry(image_slice, aperture)
                flux = phot["aperture_sum"][0]
                err = 1.
            
            
            annulus_mask = annulus.to_mask(method="center")
            annulus_data = annulus_mask.multiply(image_slice)
            annulus_data = annulus_data[annulus_data != 0]
            sky_median = np.nanmedian(annulus_data)

            if not np.isfinite(sky_median):
                sky_median = 0.

            substracted_flux = flux #- (sky_median * aperture_area)
            calibrated = substracted_flux * calibration
            
            calibrated_err = err * calibration

            spec_flux_values.append(calibrated)
            err_values.append(calibrated_err)

        spec_final = np.array(spec_flux_values)
        err_final = np.array(err_values)

        return wl_grid, spec_final, err_final

    def find_line_region(self, search_halfwidth=30.0):

        search_window = (self.wave > self.lamda_center - search_halfwidth) & \
                        (self.wave < self.lamda_center + search_halfwidth)

        if not np.any(search_window):
            return np.nan, np.nan, np.nan

        sub_wave = self.wave[search_window]
        sub_spec = self.spec[search_window]

        noise_rms = np.nanstd(self.spec[~search_window])
        cont_level = np.nanmedian(self.spec[~search_window])
        threshold = cont_level + self.detect_snr_threshold * noise_rms

        smoothed = gaussian_filter1d(sub_spec, sigma=1.0)
        
        peaks, properties = find_peaks(smoothed, height=threshold, distance=3)

        if len(peaks) == 0:
            return np.nan, np.nan, np.nan

        best_peak_local = peaks[np.argmax(smoothed[peaks])]
        peak_wave = sub_wave[best_peak_local]
        peak_idx = np.where(self.wave == peak_wave)[0][0]

        left = peak_idx
        while left > 0 and self.spec[left] > cont_level + 0.5 * noise_rms:
            left -= 1
        right = peak_idx
        while right < len(self.spec) - 1 and self.spec[right] > cont_level + 0.5 * noise_rms:
            right += 1

        return peak_wave, self.wave[left], self.wave[right]

    def fit_model(self):
        
        empty = dict(
            flux=np.nan, flux_trapz=np.nan, cont=np.nan, line_mask=np.nan,
            center=np.nan, popt=np.nan, amp_err=np.nan, chi2_red=np.nan,
            fit_significant=False, fwhm_kms_err_ = np.nan,
                        z_err_ = np.nan
        )

        peak_wave, line_min, line_max = self.find_line_region()

        line_indizes = (self.wave > line_min) & (self.wave < line_max)
        cont_indizes = ((self.wave > line_min-20) & (self.wave<line_min)) | ((self.wave>line_max) & (self.wave<line_max+20))

        if np.isnan(peak_wave):
            return empty

        rough_width = max(line_max - line_min, 2.0)
        measured_width = line_max - line_min
        
        sigma_guess = np.clip(measured_width / 5.0, 1., 15.0)

        amp_guess = np.nanmax(self.spec[line_indizes]) #- np.nanmedian(self.spec[line_indizes])
        cont_guess = np.nanmedian(self.spec[cont_indizes])

        p0 = [amp_guess, peak_wave, sigma_guess, cont_guess]

        bounds = (
            [amp_guess*0.95, line_min, 0.8, -np.inf],
            [amp_guess, line_max, sigma_guess*1.5, np.inf],
        )

        try:
            popt, pcov = curve_fit(gaussian, self.wave, self.spec, sigma=self.error, absolute_sigma=True, p0=p0, bounds=bounds, maxfev=5000)
            amp, mu, sigma, cont = popt
            perr = np.sqrt(np.diag(pcov))
            amp_err = perr[0]

            fit_line_min = mu - 3 * sigma
            fit_line_max = mu + 3 * sigma
            line_mask = (self.wave > fit_line_min) & (self.wave < fit_line_max)

            flux_trapz = np.trapezoid(self.spec[line_mask] - cont, self.wave[line_mask])
            flux_gauss = amp * abs(sigma) * np.sqrt(2 * np.pi)

            search_window = (self.wave > self.lamda_center - 20) & (self.wave < self.lamda_center + 20)
            noise_rms = np.nanstd(self.spec[~search_window]) if np.any(~search_window) else np.nan

            dof = int(np.sum(line_mask)) - len(popt)
            if dof > 0 and np.isfinite(noise_rms) and noise_rms > 0:
                model_vals = gaussian(self.wave[line_mask], *popt)
                chi2 = np.nansum(((self.spec[line_mask] - model_vals) / noise_rms) ** 2)
                chi2_red = chi2 / dof
            else:
                chi2_red = np.nan

            fit_significant = bool(
                np.isfinite(amp_err) and amp_err > 0 and (amp / amp_err) >= self.min_amp_snr
            )

        except RuntimeError:
            return empty

        r_spec=800

        perr = np.sqrt(np.diag(pcov))
        mu_err = perr[1]
        sigma_err = perr[2]

        fwhm_obs = 2.3548 * sigma
        fwhm_inst_AA = mu / r_spec

        try:
            if pcov is not None and np.all(np.isfinite(pcov)):
                diag = np.diag(pcov)
                if np.all(diag >= 0):
                    perr = np.sqrt(diag)
                    mu_err = perr[1]
                    sigma_err = perr[2]

                    redshift_err = mu_err / 1215.670

                    fwhm_obs = 2.3548 * sigma
                    fwhm_inst_AA = mu / r_spec
                    if fwhm_obs > fwhm_inst_AA:
                        fwhm_intrinsic_AA = np.sqrt(fwhm_obs**2 - fwhm_inst_AA**2)
                        dfwhm_dsigma = (2.3548**2 * sigma) / fwhm_intrinsic_AA
                        err_intrinsic = abs(dfwhm_dsigma) * sigma_err
                        c_kms = 299792.458
                        fwhm_kms_err = (err_intrinsic / mu) * c_kms
        except (ValueError, ZeroDivisionError, IndexError):
            fwhm_kms_err = np.nan
            redshift_err = np.nan


        return dict(
            flux=flux_gauss,
            flux_trapz=flux_trapz,
            cont=cont,
            line_mask=line_mask,
            center=mu,
            popt=popt,
            amp_err=amp_err,
            chi2_red=chi2_red,
            fit_significant=fit_significant,
            fwhm_kms_err_ = fwhm_kms_err,
            z_err_ = redshift_err
        )

    def mc_flux_err(self, n_iter=10):

        if _is_nan_scalar(self.line_mask):
            return np.nan, np.nan, np.nan, np.nan, 0.0

        noise_rms = np.nanstd(self.spec[~self.line_mask])
        center_idx = np.argmin(np.abs(self.wave - self.center))
        fluxes = []
        conts = []
        centers = []
        widths = []
        n_success = 0

        popt_init = self.popt

        for _ in range(n_iter):
            perturbed = self.spec + np.random.normal(0, noise_rms, size=self.spec.shape)
            try:
                amp_guess = perturbed[center_idx] - np.nanmedian(perturbed)
                p0 = amp_guess, self.center, popt_init[2], np.nanmedian(perturbed)
                bounds = (
                    [-np.inf, self.center - 20, popt_init[2]*0.8, -np.inf],
                    [np.inf, self.center + 20, popt_init[2]*2, np.inf],
                )
                popt, _ = curve_fit(gaussian, self.wave, perturbed, p0, bounds=bounds, maxfev=5000)
                amp, center, sigma, cont = popt
                fluxes.append(amp * abs(sigma) * np.sqrt(2 * np.pi))
                conts.append(cont)
                centers.append(center)
                widths.append(sigma)
                n_success += 1
            except RuntimeError:
                continue

        success_rate = n_success / n_iter

        if success_rate < self.min_mc_success_rate:
            return np.nan, np.nan, np.nan, np.nan, success_rate

        fluxes_arr = np.array(fluxes)
        conts_arr = np.array(conts)
        centers = np.array(centers)
        widths = np.array(widths)
        return np.std(fluxes_arr), np.std(conts_arr), np.std(centers), np.std(widths), success_rate

    def fwhm(self, r_spec=800):
        if _is_nan_scalar(self.popt):
            return np.nan, np.nan

        _, mu, sigma, _ = self.popt
        w_err = self.width_err
        fwhm_obs = 2.3548 * sigma

        fwhm_inst_AA = mu / r_spec
        fwhm_intrinsic_AA = np.sqrt(max(fwhm_obs ** 2 - fwhm_inst_AA ** 2, 0))

        err_intrinsic = 2.3548**2 * sigma * w_err / fwhm_intrinsic_AA

        c_kms = 299792.458
        return fwhm_intrinsic_AA / mu * c_kms, err_intrinsic / mu * c_kms

    def snr(self):
        if np.isnan(self.peak_flux) or np.isnan(self.flux_err) or self.flux_err == 0:
            return np.nan
        return self.peak_flux / self.flux_err

    def get_g_band_mag(self, tol=1.):
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
        f_lambda = 10 ** (-0.4 * (g_mag + 48.6)) * c / lam_eff ** 2
        f_cont = (f_lambda - self.peak_flux / band_width) * corr

        cont_err = np.sqrt(
            (self.flux_err / band_width) ** 2 +
            (f_lambda * np.log(10) * 0.4 * self.mag_err) ** 2
        )

        return f_cont, cont_err

    def get_cont(self):
        if np.isnan(self.g_band_mag):
            cont, err = self.cont_fit, self.cont_err_raw
        else:
            cont, err = self.cont_hsc()

        if np.isnan(cont) or cont <= 0:
            if not _is_nan_scalar(self.line_mask):
                noise = np.nanstd(self.spec[~self.line_mask])
                return noise, noise, True
            else:
                return np.nan, np.nan, True
        return cont, err, False

    def ew(self):
        if np.isnan(self.peak_flux) or np.isnan(self.cont) or self.cont == 0:
            return np.nan, np.nan, np.nan

        ew_obs = self.peak_flux / self.cont
        ew = ew_obs / (1 + self.z)

        try:
            rel_err_sq = (self.flux_err / self.peak_flux) ** 2 + (self.cont_err / self.cont) ** 2
            err = ew_obs * np.sqrt(rel_err_sq)
        except (ZeroDivisionError, TypeError, ValueError):
            err = np.nan

        return ew_obs, ew, err

    def measure_ew(self):
        ew_obs, ew, err = self.ew_obs, self.ew, self.ew_err
        flux = self.peak_flux
        cont = self.cont
        z = self.redshift
        flux_err = self.flux_err
        cont_err = self.cont_err

        return (
            ew_obs, ew, err, flux, flux_err, cont, cont_err, z, self.z_err, self.fwhm_kms, self.fwhm_kms_err, self.snr_,
            self.detected, self.fit_significant, self.chi2_red,
            self.mc_success_rate, self.cont_is_noise_floor,
        )

    def plot_ew(self, ax=None, save_path=None, show=False):
        
        if ax is None:
            fig, current_ax = plt.subplots(figsize=(3.051, 3.051))
        else:
            current_ax = ax
            fig = current_ax.figure

        if _is_nan_scalar(self.popt):
            wave, spec = self.wave, self.spec
            current_ax.plot(wave, spec, color="blue", lw=1, label="Spectrum")
            current_ax.set_xlabel(r"$\lambda [\mathrm{\AA}]$", fontsize=11)
            current_ax.set_ylabel(r"$f_{\mathrm{Ly}\alpha} [\mathrm{erg \, s^{-1} \, cm^{-2} \, \mathrm{\AA}^{-1}}]$", fontsize=10)

        else:
            ew = self.ew
            wave = self.wave
            cont = self.cont / 1e-17
            amp, mu, sig, con = self.popt
            spec = self.spec / 1e-17
            gauss = gaussian(wave, amp, mu, sig, con) / 1e-17
            line_mask = self.line_mask

            plot_mask = (wave > (mu - 80)) & (wave < (mu + 80))

            current_ax.plot(wave[plot_mask], spec[plot_mask], color="blue", lw=1, label="Spectrum")
            current_ax.plot(wave[plot_mask], gauss[plot_mask], color="red", lw=2, ls=":", alpha=0.5, label="Gauss Fit")
            current_ax.fill_between(wave[line_mask], spec[line_mask], 0, color="grey", alpha=0.3, label="Line Region")
            current_ax.axhline(y=cont, color="green", ls="--", lw=1, label="Cont Level")
            current_ax.set_xlabel(r"$\lambda [\mathrm{\AA}]$", fontsize=11)
            current_ax.set_ylabel(r"$f_{\mathrm{Ly}\alpha} [10^{-17} \, \mathrm{erg \, s^{-1} \, cm^{-2} \, \mathrm{\AA}^{-1}}]$", fontsize=11)
            #current_ax.legend(loc="best", fontsize=8)
            title = rf"$z$ = {self.redshift-0.002:.3f}"
            current_ax.set_title(title, fontsize=11)

        if ax is None:
            plt.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            
        if show:
            plt.show()

        if ax is None:
            plt.close(fig)