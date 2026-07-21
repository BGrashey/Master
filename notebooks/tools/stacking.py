import numpy as np
import zarr

from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats, sigma_clip
from astropy.io import fits

from astropy.cosmology import Planck18
import astropy.units as u

from reproject import reproject_exact

from photutils.aperture import (
    CircularAperture,
    aperture_photometry,
    CircularAnnulus,
    ApertureStats,
)
from photutils.centroids import centroid_com

import warnings

#with fits.open("/Users/bene/Desktop/mpe/fits/cubes/header.fits") as h:
#    header = h[0].header
with fits.open("/data/hetdex/u/bgrashey/cubes/ssa22_fullfp_stack.fits") as h:
    header = h[0].header
wcs = WCS(header)

COLNAMES = {
    "ra":  ["ra", "RA", "Ra", "RAJ2000", "ra_vdfi", "ra_hetdex"],
    "dec": ["dec", "DEC", "Dec", "DEJ2000", "dec_vdfi", "dec_hetdex"],
    "z":   ["z", "Z", "redshift", "REDSHIFT", "zspec", "ZSPEC", "z_vdfi", "z_hetdex", "redshift"],
    "flux":    ["flux", "Flux", "FLUX", "flux_lya"],
    "luminosity": ["lum", "luminosity", "luminosity_lae", "LUMINOSITY"],
    "completeness": ["completeness", "comp", "COMPLETENESS"],
}
 
def _find_col(table, aliases):
    for name in aliases:
        if name in table.colnames:
            return name
    raise KeyError(f"Keine Spalte gefunden. Erwartet: {aliases} | Vorhanden: {table.colnames}")

def make_aperture_mask(shape, center, radius, ellipse_axes=None, theta=0.0):
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    yc, xc = center
 
    dx = xx - xc
    dy = yy - yc
 
    if ellipse_axes is not None:
        a, b = ellipse_axes
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        x_rot = dx * cos_t + dy * sin_t
        y_rot = -dx * sin_t + dy * cos_t
        mask = (x_rot / a) ** 2 + (y_rot / b) ** 2 <= 1.0
    else:
        mask = dx**2 + dy**2 <= radius**2
 
    return mask

def subtract_sky_sigmaclip(cube, mask=None, sigma=3.0, maxiters=5, stat="median"):

    data = np.asarray(cube, dtype=float)
    n_wave, ny, nx = data.shape
 
    if mask is None:
        excl = np.zeros((ny, nx), dtype=bool)
        excl_is_3d = False
    else:
        mask = np.asarray(mask, dtype=bool)
        excl_is_3d = mask.ndim == 3
        excl = mask
 
    sky_sub_cube = np.empty_like(data)
 
    for k in range(n_wave):
        sl = data[k]
        excl_k = excl[k] if excl_is_3d else excl
 
        valid = np.isfinite(sl) & (~excl_k)
        if not np.any(valid):
            sky_sub_cube[k] = sl
            continue
 
        mean_c, median_c, std_c = sigma_clipped_stats(sl[valid], sigma=sigma, maxiters=maxiters)
        sky_level = median_c if stat == "median" else mean_c
        sky_sub_cube[k] = sl - sky_level
 
    return sky_sub_cube
 
def prepare_subcube(ra, dec, z, zarr_cube, width=25, spec_width=25):
    spec = (z + 1) * 1216  # Å
    x, y, z_pix = wcs.all_world2pix(ra, dec, spec, 0)
    xi, yi, zi = int(round(float(x))), int(round(float(y))), int(round(float(z_pix)))

    n_wave, ny, nx = zarr_cube.shape

    z0, z1 = zi - spec_width, zi + spec_width
    y0, y1 = yi - width, yi + width
    x0, x1 = xi - width, xi + width

    out_of_bounds = (z0 < 0 or z1 > n_wave or y0 < 0 or y1 > ny or x0 < 0 or x1 > nx)
    """
    if not out_of_bounds:
        sub_cube = np.asarray(zarr_cube[z0:z1, y0:y1, x0:x1])
    else:
        z0c, z1c = max(z0, 0), min(z1, n_wave)
        y0c, y1c = max(y0, 0), min(y1, ny)
        x0c, x1c = max(x0, 0), min(x1, nx)

        valid_chunk = np.asarray(zarr_cube[z0c:z1c, y0c:y1c, x0c:x1c])

        full_shape = (2 * spec_width, 2 * width, 2 * width)
        sub_cube = np.full(full_shape, np.nan, dtype=float)

        zi0, zi1 = z0c - z0, z0c - z0 + (z1c - z0c)
        yi0, yi1 = y0c - y0, y0c - y0 + (y1c - y0c)
        xi0, xi1 = x0c - x0, x0c - x0 + (x1c - x0c)

        sub_cube[zi0:zi1, yi0:yi1, xi0:xi1] = valid_chunk

    sub_wcs = wcs.deepcopy()
    sub_wcs.wcs.crpix[0] -= x0
    sub_wcs.wcs.crpix[1] -= y0
    sub_wcs.wcs.crpix[2] -= z0

    return sub_cube, sub_wcs.celestial
    """
    
    if out_of_bounds:
        return None, None   # Quelle liegt nicht vollständig im Cube

    sub_cube = np.asarray(zarr_cube[z0:z1, y0:y1, x0:x1])

    sub_wcs = wcs.deepcopy()
    sub_wcs.wcs.crpix[0] -= x0
    sub_wcs.wcs.crpix[1] -= y0
    sub_wcs.wcs.crpix[2] -= z0

    return sub_cube, sub_wcs.celestial
    

def subtract_sky_per_slice(subcube):

    ny, nx = subcube.shape[1], subcube.shape[2]

    mask = make_aperture_mask(
        shape=(ny, nx),
        center=(ny//2, nx//2),
        radius=ny//4
        )
    
    sky_substracted = subtract_sky_sigmaclip(
        subcube,
        mask=mask,
        sigma=3.,
        maxiters=5,
        stat="median",
    )

    return sky_substracted

def subtract_continuum(cube, line_mask=None, degree=2, sigma=3.0, maxiters=5):
    n_wave, ny, nx = cube.shape
    wave = np.arange(n_wave)
    if line_mask is None:
        line_mask = np.zeros(n_wave, dtype=bool)
        center = n_wave // 2
        line_mask[center-9:center+10] = True

    continuum = np.full_like(cube, np.nan)

    for j in range(ny):
        for i in range(nx):
            spec = cube[:, j, i]
            valid = np.isfinite(spec) & (~line_mask)
            if np.sum(valid) < degree + 1:
                continue

            x_fit, y_fit = wave[valid], spec[valid]

            # iteratives Sigma-Clipping auf die Residuen
            for _ in range(maxiters):
                coeffs = np.polyfit(x_fit, y_fit, deg=degree)
                resid = y_fit - np.polyval(coeffs, x_fit)
                clipped = sigma_clip(resid, sigma=sigma, maxiters=1)
                keep = ~clipped.mask
                if keep.sum() == len(x_fit):
                    break
                x_fit, y_fit = x_fit[keep], y_fit[keep]

            continuum[:, j, i] = np.polyval(coeffs, wave)

    cont_sub_cube = cube - continuum
    return cont_sub_cube
    
def make_narrowband(subcube, line_mask=None):

    n_wave, _, _ = subcube.shape
    wave = np.arange(n_wave)
    if line_mask is None:
        line_mask = np.zeros(n_wave, dtype=bool)
        center = n_wave // 2
        line_mask[center-9:center+10] = True
        
    selected = subcube[line_mask]
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        image = np.nanmean(selected, axis=0)

    return image

def make_wcs(ra, dec, z, kpc_per_pixel=3, npix=50):
    kpc_per_arcsec = Planck18.kpc_proper_per_arcmin(z).to(u.kpc / u.arcsec).value
    arcsec_per_pixel = kpc_per_pixel / kpc_per_arcsec
    deg_per_pixel = arcsec_per_pixel / 3600.0

    w = WCS(naxis=2)
    w.wcs.crpix = [npix / 2, npix / 2]
    w.wcs.cdelt = [-deg_per_pixel, deg_per_pixel]
    w.wcs.crval = [ra, dec]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w

def scale_source(img, wcs, ra, dec, z, kpc_per_pixel=3., npix=50):
    target_wcs = make_wcs(ra, dec, z, kpc_per_pixel, npix)
    regrid, footprint = reproject_exact(
        (img, wcs), target_wcs, shape_out=(npix, npix)
    )

    return regrid, footprint

"""
def stacking(catalog, cube, width=25, spec_width=25, kpc_pxl=3, npix=50):
    col_ra = _find_col(catalog, COLNAMES["ra"])
    col_dec = _find_col(catalog, COLNAMES["dec"])
    col_z = _find_col(catalog, COLNAMES["z"])

    images = []
    footprints = []

    for i in range(len(catalog)):
        ra, dec, z = catalog[i][col_ra], catalog[i][col_dec], catalog[i][col_z]
        subcube, sub_wcs = prepare_subcube(ra, dec, z, cube, width=width, spec_width=spec_width)
        subtracted = subtract_sky_per_slice(subcube)
        contsub = subtract_continuum(subtracted)
        img = make_narrowband(contsub)
        scaled, foot = scale_source(img, sub_wcs, ra, dec, z, kpc_per_pixel=kpc_pxl, npix=npix)
        images.append(scaled)
        footprints.append(foot)

    stack_data = np.array(images)
    stack_footprint = np.array(footprints)

    weighted_stack = np.nansum(stack_data * stack_footprint, axis=0) / np.nansum(stack_footprint, axis=0)
    
    return weighted_stack
def cog(img, r_max=12, kpc_pxl=3):
        
        data_slice = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

        x0, y0 = centroid_com(data_slice)

        radii = np.arange(2, r_max, 1)
        apertures = [
            CircularAperture((x0, y0), r=r) for r in radii
        ]

        fluxes = []

        for ap in apertures:
            phot = aperture_photometry(data_slice, ap)
            aperture_flux = phot["aperture_sum"][0]
            fluxes.append(aperture_flux)

        cum_flux = np.array(fluxes)
        flux_arr = np.diff(cum_flux, prepend=0.0)
        radii_kpc = radii*kpc_pxl

        return radii_kpc, flux_arr, cum_flux, x0, y0
"""


class Stacking:
    """
    Class to perform stacking of lyman alpha emitters. It needs only a astropy
    Table and a loaded zarr cube to perform the stacking.
    
    Parameters
    catalog: astropy Table object
    cube: loaded zarr array
    width: width of the cutout
    spec_width: width of the cutout spectrally
    kpc_pxl: how many kpc should a pixel in the stack have
    npix: width of the stacked image
    """

    def __init__(
            self,
            catalog,
            cube,
            width=25,
            spec_width=25,
            kpc_pxl=3,
            npix=50,
    ):
        self.catalog = catalog
        self.cube = cube
        self.width = width
        self.spec_width = spec_width
        self.kpc_pxl = kpc_pxl
        self.npix = npix


    def stack(self):
        col_ra = _find_col(self.catalog, COLNAMES["ra"])
        col_dec = _find_col(self.catalog, COLNAMES["dec"])
        col_z = _find_col(self.catalog, COLNAMES["z"])

        images = []
        footprints = []
        
        n_skipped = 0

        for i in range(len(self.catalog)):
            ra, dec, z = self.catalog[i][col_ra], self.catalog[i][col_dec], self.catalog[i][col_z]
            subcube, sub_wcs = prepare_subcube(ra, dec, z, self.cube, width=self.width, spec_width=self.spec_width)
            
            if subcube is None:
                n_skipped += 1
                continue
            
            subtracted = subtract_sky_per_slice(subcube)
            contsub = subtract_continuum(subtracted)
            img = make_narrowband(contsub)
            scaled, foot = scale_source(img, sub_wcs, ra, dec, z, kpc_per_pixel=self.kpc_pxl, npix=self.npix)
            images.append(scaled)
            footprints.append(foot)

        stack_data = np.array(images)
        stack_footprint = np.array(footprints)

        weighted_stack = np.nansum(stack_data * stack_footprint, axis=0) / np.nansum(stack_footprint, axis=0)
        
        print(f"Skipped: {n_skipped}")
        
        return weighted_stack
    
    def cog(self, img, r_max=12):
        
        data_slice = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

        x0, y0 = centroid_com(data_slice)

        radii = np.arange(2, r_max, 1)
        apertures = [
            CircularAperture((x0, y0), r=r) for r in radii
        ]

        fluxes = []

        for ap in apertures:
            phot = aperture_photometry(data_slice, ap)
            aperture_flux = phot["aperture_sum"][0]
            fluxes.append(aperture_flux)

        cum_flux = np.array(fluxes)
        flux_arr = np.diff(cum_flux, prepend=0.0)
        radii_kpc = radii*self.kpc_pxl

        return radii_kpc, flux_arr, cum_flux