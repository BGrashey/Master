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
    return cont + amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _is_nan_scalar(value):
    """Konsistenter Check fuer die 'nicht gemessen'-Sentinel (np.nan) die im
    Rest der Klasse als Ersatz fuer None/Optional benutzt wird."""
    return np.isscalar(value) and np.isnan(value)


class Measurements:
    """
    Class to perform spectral and spatial measurements on 3D data cubes.

    Qualitaets-/Detektions-Flags (wichtig bei verrauschten Spektren):
        self.detected            : bool - wurde ueberhaupt eine Linie ueber
                                    der Rausch-Schwelle gefunden UND war der
                                    Gauss-Fit signifikant (amp/amp_err)?
        self.fit_significant      : bool - Amplitude des Gauss-Fits signifikant
                                    ueber ihrem eigenen Fehler (aus pcov)?
        self.chi2_red             : reduziertes Chi^2 des Gauss-Fits im
                                    line_mask-Bereich (Diagnose fuer schlechte
                                    Fits, die formal trotzdem konvergiert sind)
        self.mc_success_rate      : Anteil der Monte-Carlo-Iterationen in
                                    mc_flux_err(), die erfolgreich konvergiert
                                    sind. Bei niedriger Rate werden flux_err/
                                    cont_err_raw auf NaN gesetzt statt eines
                                    kuenstlich zu kleinen Fehlers aus wenigen
                                    "guenstigen" Realisierungen.
        self.cont_is_noise_floor  : bool - self.cont ist KEIN echtes
                                    Kontinuum, sondern ein Rauschniveau-
                                    Platzhalter (Kontinuum-Fit/HSC-Photometrie
                                    war nicht verfuegbar/negativ). EW-Werte,
                                    die darauf basieren, sind effektiv nur
                                    obere Limits und sollten entsprechend
                                    markiert/behandelt werden.

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
        if self.data.size == 0:
            raise ValueError(f"Object at ({self.ra}, {self.dec}) is fully outside cube bounds.")

        if self.data.shape != (2 * pad_lam, 2 * pad_xy, 2 * pad_xy):
            import warnings
            warnings.warn(
                f"Sub-cube is truncated at the edge: shape={self.data.shape}. "
                "Results may be less reliable.",
                RuntimeWarning
            )

        self.catalog = catalog
        self.catalog_coord = catalog_skycoord

        self.detect_snr_threshold = detect_snr_threshold
        self.min_amp_snr = min_amp_snr
        self.min_mc_success_rate = min_mc_success_rate

        self.wave, self.spec = self.get_spectrum()

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

        self.flux_err, self.cont_err_raw, self.mc_success_rate = self.mc_flux_err()
        self.fwhm_kms = self.fwhm()
        self.snr_ = self.snr()
        self.g_band_mag, self.mag_err = self.get_g_band_mag()
        self.cont, self.cont_err, self.cont_is_noise_floor = self.get_cont()
        self.ew_obs, self.ew, self.ew_err = self.ew()
        self.redshift = self.center / 1215.670 - 1 if not _is_nan_scalar(self.center) else np.nan

        # Gesamtstatus: nur wenn wir ueberhaupt eine Linie ueber der
        # Rausch-Schwelle gefunden haben UND der Fit signifikant war.
        self.detected = bool(self.fit_significant) and not _is_nan_scalar(self.peak_flux)

    # ---------------------------------------------------------------------
    # Measurement functions
    # ---------------------------------------------------------------------
    def cog(self, r_max=15, threshold=0.05, n_avg_channels=5):
        """
        Perform a curve of growth to find the necessary aperture.

        Um die COG-Bestimmung bei verrauschten Daten nicht von einer
        einzelnen, verrauschten Wellenlaengen-Scheibe abhaengig zu machen,
        wird ueber n_avg_channels Kanaele um die geschaetzte Linienmitte
        gemittelt (statt nur self.data[self.center_idx]).
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

        for i in range(len(wl_grid)):
            image_slice = self.data[i, :, :]
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

    def find_line_region(self, smooth_sigma=2, search_halfwidth=20):
        """
        Sucht den staerksten Peak im Suchfenster um lamda_center und gibt
        seine Grenzen zurueck - ABER nur, wenn der Peak tatsaechlich ueber
        self.detect_snr_threshold * noise_rms liegt. Andernfalls wird
        (nan, nan, nan) zurueckgegeben ("keine Linie gefunden"), statt
        einen Rauschpeak als Kandidaten weiterzureichen.
        """
        smoothed = gaussian_filter1d(self.spec, sigma=smooth_sigma)

        search_window = (self.wave > self.lamda_center - search_halfwidth) & \
                        (self.wave < self.lamda_center + search_halfwidth)

        if not np.any(search_window):
            return np.nan, np.nan, np.nan

        noise_rms = np.nanstd(self.spec[~search_window])
        cont_level = np.nanmedian(self.spec[~search_window])
        threshold = cont_level + self.detect_snr_threshold * noise_rms

        idx_in_window = np.where(search_window)[0]
        peak_idx = idx_in_window[np.argmax(smoothed[search_window])]
        peak_wave = self.wave[peak_idx]

        # Detektions-Gate: ohne signifikanten Peak keine "gefundene" Linie.
        if not np.isfinite(smoothed[peak_idx]) or smoothed[peak_idx] < threshold:
            return np.nan, np.nan, np.nan

        left = peak_idx
        while left > 0 and smoothed[left] > threshold:
            left -= 1
        right = peak_idx
        while right < len(smoothed) - 1 and smoothed[right] > threshold:
            right += 1

        return peak_wave, self.wave[left], self.wave[right]

    def fit_model(self):
        """
        Gauss-Fit an die Linie. Liefert ein dict mit allen relevanten
        Groessen UND Qualitaets-Flags (amp_err, chi2_red, fit_significant),
        damit ein formal konvergierter, aber physikalisch bedeutungsloser
        Fit an Rauschen erkennbar bleibt.

        Als 'flux' wird konsistent der ANALYTISCHE Gauss-Flux
        (amp * sigma * sqrt(2*pi)) verwendet - dieselbe Groesse, die auch
        in mc_flux_err() fuer den MC-Fehler benutzt wird. Der rohe
        Trapez-Fluss der (verrauschten) Daten wird zusaetzlich als
        Diagnosewert unter 'flux_trapz' mitgegeben, ist aber nicht mehr
        der primaere Schaetzer.
        """
        empty = dict(
            flux=np.nan, flux_trapz=np.nan, cont=np.nan, line_mask=np.nan,
            center=np.nan, popt=np.nan, amp_err=np.nan, chi2_red=np.nan,
            fit_significant=False,
        )

        peak_wave, line_min, line_max = self.find_line_region()

        if np.isnan(peak_wave):
            return empty

        rough_width = max(line_max - line_min, 2.0)
        sigma_guess = rough_width / 4.0

        amp_guess = np.nanmax(self.spec) - np.nanmedian(self.spec)
        amp_max = max(3 * amp_guess, 1e-19)
        cont_guess = np.nanmedian(self.spec)

        p0 = [amp_guess, peak_wave, sigma_guess, cont_guess]

        bounds = (
            [0, line_min - 3, 0.5, -np.inf],
            [amp_max, line_max + 3, rough_width, np.inf],
        )

        try:
            popt, pcov = curve_fit(gaussian, self.wave, self.spec, p0=p0, bounds=bounds, maxfev=5000)
            amp, mu, sigma, cont = popt
            perr = np.sqrt(np.diag(pcov))
            amp_err = perr[0]

            fit_line_min = mu - 3 * sigma
            fit_line_max = mu + 3 * sigma
            line_mask = (self.wave > fit_line_min) & (self.wave < fit_line_max)

            flux_trapz = np.trapezoid(self.spec[line_mask] - cont, self.wave[line_mask])
            flux_gauss = amp * abs(sigma) * np.sqrt(2 * np.pi)

            # Guete des Fits: reduziertes Chi^2 im Linienbereich, geschaetzt
            # ueber das Rausch-RMS ausserhalb des Fensters um lamda_center.
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
        )

    def mc_flux_err(self, n_iter=200):
        """
        Monte-Carlo-Fehlerschaetzung fuer flux und Kontinuum durch
        wiederholtes Fitten von rauschperturbierten Spektren.

        Aenderungen ggue. vorher:
        - p0/bounds werden um das GEFITTETE Linienzentrum (self.center)
          statt um das rein systemische lamda_center gebaut, da Lyα haeufig
          gegenueber dem systemischen z verschoben ist - das verbessert die
          Konvergenzrate gerade bei verrauschten Spektren.
        - Es wird mitgezaehlt, wie viele der n_iter Iterationen tatsaechlich
          konvergieren. Liegt die Erfolgsquote unter
          self.min_mc_success_rate, gelten die Fehler als nicht
          verlaesslich und werden auf NaN gesetzt (statt still aus wenigen
          "guenstigen" Realisierungen einen zu kleinen Fehler zu bekommen).
        """
        if _is_nan_scalar(self.line_mask):
            return np.nan, np.nan, 0.0

        noise_rms = np.nanstd(self.spec[~self.line_mask])
        center_idx = np.argmin(np.abs(self.wave - self.center))
        fluxes = []
        conts = []
        n_success = 0

        for _ in range(n_iter):
            perturbed = self.spec + np.random.normal(0, noise_rms, size=self.spec.shape)
            try:
                amp_guess = perturbed[center_idx] - np.nanmedian(perturbed)
                p0 = amp_guess, self.center, 2, np.nanmedian(perturbed)
                bounds = (
                    [-np.inf, self.center - 20, 0.5, -np.inf],
                    [np.inf, self.center + 20, 15, np.inf],
                )
                popt, _ = curve_fit(gaussian, self.wave, perturbed, p0, bounds=bounds, maxfev=5000)
                amp, _, sigma, cont = popt
                fluxes.append(amp * abs(sigma) * np.sqrt(2 * np.pi))
                conts.append(cont)
                n_success += 1
            except RuntimeError:
                continue

        success_rate = n_success / n_iter

        if success_rate < self.min_mc_success_rate:
            return np.nan, np.nan, success_rate

        fluxes_arr = np.array(fluxes)
        conts_arr = np.array(conts)
        return np.std(fluxes_arr), np.std(conts_arr), success_rate

    def fwhm(self, r_spec=750):
        if _is_nan_scalar(self.popt):
            return np.nan

        _, mu, sigma, _ = self.popt
        fwhm_obs = 2.3548 * sigma

        fwhm_inst_AA = mu / r_spec
        fwhm_intrinsic_AA = np.sqrt(max(fwhm_obs ** 2 - fwhm_inst_AA ** 2, 0))

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
        f_lambda = 10 ** (-0.4 * (g_mag + 48.6)) * c / lam_eff ** 2
        f_cont = (f_lambda - self.peak_flux / band_width) * corr

        cont_err = np.sqrt(
            (self.flux_err / band_width) ** 2 +
            (f_lambda * np.log(10) * 0.4 * self.mag_err) ** 2
        )

        return f_cont, cont_err

    def get_cont(self):
        """
        Rueckgabe: (cont, cont_err, cont_is_noise_floor)

        cont_is_noise_floor=True bedeutet: weder der Kontinuum-Fit noch die
        HSC-Photometrie lieferten einen brauchbaren (positiven, endlichen)
        Wert - stattdessen wurde das Rausch-RMS ausserhalb der Linie als
        Platzhalter benutzt. EW-Werte auf dieser Basis sind effektiv nur
        obere Limits und sollten in nachgelagerten Analysen entsprechend
        gefiltert/markiert werden.
        """
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
            ew_obs, ew, err, flux, flux_err, cont, cont_err, z, self.fwhm_kms, self.snr_,
            self.detected, self.fit_significant, self.chi2_red,
            self.mc_success_rate, self.cont_is_noise_floor,
        )

    def plot_ew(self, save_path=None, show=False):
        """
        save_path : str oder None — falls gesetzt, wird der Plot dort gespeichert
        show      : bool — ob der Plot interaktiv angezeigt werden soll (nur für Einzelfälle sinnvoll)
        """
        if _is_nan_scalar(self.popt):
            raise RuntimeError(
                "Kein signifikanter Fit vorhanden (self.detected=False) - "
                "plot_ew() kann kein Modell zeichnen."
            )

        ew = self.ew
        spec = self.spec
        wave = self.wave
        cont = self.cont
        amp, mu, sig, con = self.popt
        gauss = gaussian(wave, amp, mu, sig, con)
        line_mask = self.line_mask

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(wave, spec, color="blue", lw=1, label="Flux")
        ax.plot(wave, gauss, color="red", lw=2, ls=":", alpha=0.5, label="Gauss Fit")
        ax.fill_between(wave[line_mask], spec[line_mask], cont, color="grey", alpha=0.3, label="Line Region")
        ax.axhline(y=cont, color="green", ls="--", lw=1, label="Cont Level")
        ax.set_xlabel("Wavelength [Å]")
        ax.set_ylabel(r"Flux $\frac{erg}{s \, cm^2 \, \AA}$")
        ax.legend(loc="best")
        title = f"EW = {ew:.1f} [Å], z = {self.redshift:.3f}"
        if self.cont_is_noise_floor:
            title += "  [Cont = noise floor]"
        if not self.fit_significant:
            title += "  [NOT significant]"
        ax.set_title(title)
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()

        plt.close(fig)