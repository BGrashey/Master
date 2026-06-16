import os

import numpy as np

from matplotlib import pyplot as plt

from tools.cubes import Cube

from astropy.modeling import models, fitting
from astropy.stats import sigma_clip
from astropy.table import Table
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord

from scipy.ndimage import median_filter, gaussian_filter1d
from scipy.signal import find_peaks


class Measurements:

    def __init__(self,
                 cube,
                 coords : tuple,
                 size : float,
                 catalog = None,
                 catalog_skycoord = None,
                 out_dir : str = None,
                 obj_name : str = None,
                 controll_cube = None,
                 degree : int = 2,
                 smoothing: int = 20
                 ):
        """Initializes the Measurements class and performs initial spectral fitting.

        Args:
            cube: Load the cube with the Cube calass from tools.cubes.
            catalog: Has to be a pandas df
            coords: Tuple containing (RA, Dec, redshift). RA and Dec should be 
                strings with units (e.g., '12h30m05s').
            size: Radius of the source aperture (or tuple for x/y size).
            out_dir: Optional; Absolute path to the output directory for saving 
                plots as PDFs.
            obj_name: Optional; Unique identifier or ID of the object.
            controll_cube:
            degree: Degree of the polynomial used for the continuum fit but max 3 to 
                not get overfitting.

        Attributes:
            data (np.ndarray): The 3D data array from the FITS file.
            wcs (astropy.wcs.WCS): The World Coordinate System from the header.
            wave (np.ndarray): The wavelength axis of the extracted spectrum.
            flux (np.ndarray): The flux values of the extracted spectrum.
            line_flux (float): The integrated flux of the measured emission line.
        """

        # get the catalog
        self.catalog = catalog
        self.catalog_coord = catalog_skycoord

        # if param is used, set output dir and object name
        self.out_dir = out_dir
        self.obj_name = obj_name
        
        self.degree = degree
        self.smoothing = int(smoothing) * 2

        # get the cubes
        self.cube = cube
        self.controll_cube = controll_cube

        # get the coordinates (RA, DEC, Z)
        self.Ra, self.Dec, self.z = coords
        # get the coords in pixels in our cube
        self.x0, self.y0 = cube.world2pix(self.Ra, self.Dec)

        # get the size of the LAE for the aperture
        self.radius = 4 #np.clip(np.mean(size) / 2, 2, 4)
        
        # measure the spectrum and aperture correction
        self.wave, self.flux = cube.get_spectrum(self.x0, self.y0, self.radius)
        

        
        # get the best fitting models and parameters
        self.center, self.sigma, self.line_mask, self.cont_mask, self.peak_model, self.cont_model, self.flux_fit, self.cont_val = self.fit_model()
        
        # get the correction factor fot total flux in the line capped at 5 to eliminate extreme values
        self.ap_corr = min(self.aperture_correction(), 10)


        
        self.flux_err, self.cont_err_raw = self.error_estimation()

        # get the cont lvl or error when too small
        self.cont, self.detection, self.cont_err = self.cont_lvl()
        
        # get the flux, flux error and the aperture correction 
        self.line_flux = self.flux_fit * self.ap_corr
    
        # get the equivalent width
        self.eq_width, self.eq_err = self.equivalent_width()

        # get g_band mag and HSC equivalent width
        self.g_mag, self.g_mag_err = self.get_g_band_mag()
        self.eq_width_HSC, self.cont_controll, self.eq_HSC_err, self.cont_HSC_err = self.equivalent_width_HSC()
        
        self.cont_ratio = 1 #self.cont / self.cont_controll

        self.redshift = self.center / 1215.670 - 1

        # ----------------------------------------------------------------------
        # define all functions for measurements or plots
        # ----------------------------------------------------------------------

    def line_flux_aperture(self, radius):

        wave, flux = self.cube.get_spectrum(self.x0, self.y0, radius)

        cont = np.median(flux[self.cont_mask])

        flux_integrated = np.trapezoid(
            flux[self.line_mask] - cont,
            wave[self.line_mask]
        )

        return flux_integrated
    

    def cog(self, rmax=15):

        cont = np.median(self.cube.data[self.cont_mask, :, :], axis=0)

        line_cube = self.cube.data[self.line_mask, :, :] - cont

        line_map = np.sum(line_cube, axis=0) * 1e-17

        y, x = np.indices(line_map.shape)
        r = np.sqrt((x - self.x0)**2 + (y - self.y0)**2)

        radii = np.arange(1, rmax + 1)
        fluxes = []

        for rad in radii:
            fluxes.append(np.sum(line_map[r <= rad]))

        return radii, np.array(fluxes)

            
    def flux_tot(self, fluxes, tol=0.05):

        fluxes = np.array(fluxes)

        valid = fluxes[:-1] > 0

        diffs = np.diff(fluxes)
        rel_change = np.zeros_like(diffs)
        rel_change[valid] = np.abs(diffs[valid] / fluxes[:-1][valid])

        idx = np.where(rel_change < tol)[0]

        if len(idx) > 0:
            return fluxes[idx[0] + 1]
        else:
            return np.max(fluxes)
          
    def aperture_correction(self, rmax=15):

        _, fluxes = self.cog(rmax)
        F_tot = self.flux_tot(fluxes)

        cont = np.median(self.flux[self.cont_mask])
        flux = self.flux[self.line_mask] - cont
        wave = self.wave[self.line_mask]

        F_sn = np.trapezoid(flux, wave)

        if F_sn <= 0:
            return 1.0

        ratio = F_tot / F_sn

        if ratio < 1:
            return 1.0

        return ratio
    """
 

    
    def aperture_correction(self, r=12):
        
        F_tot = self.line_flux_aperture(r)
        F_sn = self.line_flux_aperture(self.radius)
        
        if F_sn <= 0:
            return 1.0
            
        ratio = F_tot / F_sn
        
        if ratio < 1:
            return 1.0
            
        return ratio
"""        
    #---------------------------------------------------------------------
    # now the measurements
    #---------------------------------------------------------------------

    def fit_model(self, width=100):

        smooth_len=self.smoothing
        wave = self.wave
        corr = 1
        center_guess = 1215.670 * (1 + self.z)

        fit_region = ((wave > (center_guess - 1 * width)) &
                        (wave < (center_guess + 1 * width)))
        
        if np.sum(fit_region) < 5:
            raise ValueError("Fit region too small")

        max_half = np.nanmax(self.flux[fit_region])
        
        wave_fit = wave[fit_region]
        max_index = np.argmax(self.flux[fit_region])
        center_est = wave_fit[max_index]
            
        flux_fit = self.flux[fit_region]
        wave_fit = wave[fit_region]

        polynomial = models.Polynomial1D(degree=self.degree)
            
        gauss = models.Gaussian1D(amplitude=max_half,
                                mean=center_est,
                                stddev=1.,
                                bounds={
                                    "mean":(center_est - 5, center_est + 5),
                                    "stddev": (1., 10.),
                                    "amplitude": (max_half/2, 2 * max_half)
                                    }
                                )
            
        fitter = fitting.LevMarLSQFitter()
        
        model = gauss + polynomial
            
        best_gauss = fitter(model,
                        wave_fit,
                        flux_fit,
                        filter_non_finite=True
                        )
        

        final_mu = best_gauss[0].mean.value
        final_sigma = best_gauss[0].stddev.value

        cont = best_gauss[1](wave)
        peak = best_gauss[0](wave)

        line_min = final_mu - 3 * final_sigma #2.355
        line_max = final_mu + 3 * final_sigma

        line_mask = (wave > line_min) & (wave < line_max)

        cont_mask = (
        ((wave > line_min - width) & (wave < line_min)) |
        ((wave > line_max) & (wave < line_max + width))
        )

        flux_line = np.trapezoid(
            (self.flux[line_mask] - cont[line_mask]),
            wave[line_mask]
        )
        cont_val = best_gauss[1](final_mu)

        return final_mu, final_sigma, line_mask, cont_mask, peak, cont, flux_line, cont_val
        
        
    def cont_lvl(self):

        cont = self.cont_val * self.ap_corr
        noise = np.std(self.cont_model[self.cont_mask] * self.ap_corr - self.flux[self.cont_mask] * self.ap_corr)
        
        err = self.cont_err_raw

        SN = cont / noise if noise > 0 else 0
        
        if cont > 0:
            return cont, "detected", err
        else:
            return noise, "lower Limit", noise
        
        
    def error_estimation(self, acc=100):
        
        best_fit_model = self.peak_model + self.cont_model
        center = self.center
        sigma = self.sigma
        corr = self.ap_corr
            
        residuals = self.flux[self.cont_mask] - self.cont_model[self.cont_mask]
        noise_sigma = np.nanstd(self.flux[self.cont_mask]) #np.std(residuals)
        
        if not np.isfinite(noise_sigma) or noise_sigma == 0:
            return np.nan, np.nan

        full_mask = self.cont_mask | self.line_mask

        wave = self.wave
        peak = np.max(best_fit_model)

        Fluxes = []
        Conts = []

        fitter = fitting.LevMarLSQFitter()
        rng = np.random.default_rng()

        for n in range(acc):
                
            noise = rng.normal(0, noise_sigma, size=len(wave))

            simulated_spec = best_fit_model + noise

            gauss = models.Gaussian1D(amplitude=peak,
                                    mean=center,
                                    stddev=sigma,
                                    bounds={
                                        "mean":(center - 5, center + 5),
                                        "stddev": (1., 30.),
                                        "amplitude": (0, None)
                                    }
                                    )
            polynomial = models.Polynomial1D(degree=self.degree)

            model = gauss + polynomial

            best_fit = fitter(
                    model,
                    wave[full_mask],
                    simulated_spec[full_mask],
                    filter_non_finite=True
                )

            final_mu = best_fit[0].mean.value
            final_sigma = best_fit[0].stddev.value
            final_amp = best_fit[0].amplitude.value

            cont = best_fit[1](wave)
            model_fit = best_fit(wave)

            line_min = final_mu - 2.355 * final_sigma
            line_max = final_mu + 2.355 * final_sigma

            line_mask = (wave > line_min) & (wave < line_max)

            flux = np.trapezoid((model_fit[line_mask] - cont[line_mask]),
                                   wave[line_mask])
            
            Fluxes.append(flux * corr)
            #Fluxes.append(final_amp * corr)
            Conts.append(best_fit[1](final_mu) * corr)
                  
        Fluxes_arr = np.array(Fluxes)
        std_flux = np.std(Fluxes_arr)
        Conts_arr = np.array(Conts)
        std_cont = np.std(Conts_arr)
        
        return std_flux, std_cont
    
    
    def equivalent_width(self):
        
        peak_flux = self.line_flux
        cont = self.cont
        z = self.z
        
        if not np.isfinite(cont) or cont == 0:
            return np.nan, np.nan
        
        ew_obs = peak_flux / cont
        ew = ew_obs / (1 + z)
        
        # error estimation
        flux_err = self.flux_err
        cont_err = self.cont_err
        #err = np.sqrt(
        #    (flux_err / cont / (1+z))**2 +
        #    (- peak_flux / cont**2 * cont_err / (1+z))**2
        #)
        err_rel = np.sqrt(
            (flux_err / peak_flux)**2 +
            (cont_err / cont)**2
        )
        err = ew * err_rel
        
        return ew, err
        
    
    def get_g_band_mag(self, tol_arcsec=2.):

        hsc_df = self.catalog
        hsc_coords = self.catalog_coord

        c_obj = SkyCoord(self.Ra, self.Dec, frame="icrs", 
                     unit=(None if isinstance(self.Ra, str) else u.deg))

        idx, d2d, _ = c_obj.match_to_catalog_sky(hsc_coords)

        if d2d.to(u.arcsec).value < tol_arcsec:
            return hsc_df.iloc[idx]["g_cmodel_mag"], hsc_df.iloc[idx]["g_cmodel_magerr"]
    
        return np.nan, np.nan
    
    def equivalent_width_HSC(self):

        g_mag = self.g_mag
        flux_line = self.flux_fit * self.ap_corr
        flux_err = self.flux_err
        z = self.z

        c = 2.99792458e18  # Å/s
        lam_eff = 4726   # Å
        BW = 1468        # Å
        
        corr = (self.center / lam_eff) **(-2)

        if np.isnan(g_mag):
            return np.nan, np.nan, np.nan, np.nan
        else:
            f_lambda = 10**(-0.4 * (g_mag + 48.6)) * c / lam_eff**2
            f_cont = (f_lambda - flux_line / BW) * corr
            cont_err = np.sqrt((flux_err / BW)**2 
                               + (f_lambda * np.log(10)*0.4*self.g_mag_err)**2
                              )
            

            ew = flux_line / f_cont / (1 + z)
            
            # error estimation
            err_rel = np.sqrt(
                (flux_err / flux_line)**2
                + (cont_err / f_cont)**2
            )
            
            err = ew * err_rel

            return ew, f_cont, err, cont_err
        
    def measure_ew(self):
        
        if self.eq_width_HSC is None or np.isnan(self.eq_width_HSC):
            ew = self.eq_width
            ew_err = self.eq_err
            source = "VDFI"
        else:
            ew = self.eq_width_HSC
            ew_err = self.eq_HSC_err
            source = "HSC"

        flux_line = self.line_flux
        flux_error = self.flux_err
        redshift = self.redshift
        """if not (0.5 <= abs(self.cont_ratio) <= 1.5):
            cont = np.nan
        else:
            cont = self.cont"""
        cont = self.cont
        cont_err = self.cont_err
        cont_hsc = self.cont_controll
        cont_hsc_err = self.cont_HSC_err
        
        if self.eq_width_HSC is None or np.isnan(self.eq_width_HSC):
            cont = self.cont
            cont_err = self.cont_err
        else:
            cont = self.cont_controll
            cont_err = self.cont_HSC_err

        return ew, ew_err, source, flux_line, flux_error, redshift, cont, cont_err, self.detection
    
    
    def plot_ew(self):

        ew = self.eq_width
        ew_HSC = self.eq_width_HSC
        flux = self.flux * self.ap_corr
        wave = self.wave
        cont = self.cont_model * self.ap_corr
        gauss = (self.peak_model + self.cont_model) * self.ap_corr
        plot_mask = self.cont_mask | self.line_mask
        line_mask = self.line_mask

        fig, ax = plt.subplots(figsize=(7,5))

        ax.plot(wave[plot_mask],
                flux[plot_mask],
                color="blue",
                lw=1,
                label="Flux"
        )
        ax.plot(
            wave[plot_mask],
            cont[plot_mask],
            color="black",
            lw=2,
            ls="--",
            label="Continuum fit"
        )
        ax.plot(
            wave[plot_mask],
            gauss[plot_mask],
            color="red",
            lw=2,
            ls=":",
            alpha=0.5,
            label="Gauss fit"
        )
        ax.fill_between(
            wave[line_mask],
            flux[line_mask],
            cont[line_mask],
            color="grey",
            alpha=0.3,
            label="Line region"
        )
        
        ax.axhline(y=self.cont, color='green', ls='--', lw=1, label="Cont_lvl")
        ax.axhline(y=self.cont_controll, color='orange', ls='--', lw=1, label="Cont_HSC")
        ax.set_xlabel("Wavelength [Å]")
        ax.set_ylabel(r"Flux $\frac{erg}{s \, cm^2 \, \AA}$")
        ax.legend(loc="best")
        ax.set_title(f"EW = {ew:.1f}, EW(HSC) = {ew_HSC:.1f} [Å], z = {self.redshift:.1f}, cont_ratio = {self.cont_ratio:.1f}")

        plt.tight_layout()

        if self.out_dir:
            filename = f"{self.obj_name}.pdf"
            full_path = os.path.join(self.out_dir, filename)
            
            if not os.path.exists(self.out_dir):
                os.makedirs(self.out_dir)
                
            plt.savefig(full_path)
            print(f"Plot saved as: {full_path}")

        plt.show()