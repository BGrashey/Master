import numpy as np
import zarr

from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats, sigma_clip
from astropy.io import fits

from astropy.cosmology import Planck18
import astropy.units as u

from reproject import reproject_exact as reproject_exact # instead of exact for performance or interp

from photutils.aperture import (
    CircularAperture,
    CircularAnnulus,
    aperture_photometry,
    ApertureStats,
)
from photutils.centroids import centroid_com

import warnings

# ---------------------------------------------------------------------------
# Einheiten / Kalibration des Eingangscubes
# ---------------------------------------------------------------------------
# Die Flusswerte im Eingangscube sind in 1e-17 erg/s/cm^2/AA PRO PIXEL
# angegeben. Ein Pixel entspricht 0.5" x 0.5" am Himmel.
#
# Fuer das Stacking auf ein gemeinsames physikalisches (kpc-)Gitter werden
# die Werte zunaechst in Oberflaechenhelligkeit (surface brightness, SB)
# umgerechnet, da unterschiedliche Quellen (unterschiedliches z) auf dem
# Zielraster unterschiedliche kpc/arcsec-Skalen haben. reproject_exact
# behandelt die Eingabe als Intensitaet -- nur wenn wir vorher durch die
# Pixelflaeche (in arcsec^2) teilen, bleibt diese Intensitaet unter dem
# Regridding physikalisch korrekt erhalten.
PIXEL_SCALE_ARCSEC = 0.5
PIXEL_AREA_ARCSEC2 = PIXEL_SCALE_ARCSEC ** 2  # 0.25 arcsec^2

FLUX_UNIT = "1e-17 erg / s / cm^2 / AA  (pro Pixel, {:.2f} arcsec^2)".format(
    PIXEL_AREA_ARCSEC2
)
SB_UNIT = "1e-17 erg / s / cm^2 / AA / arcsec^2"


def flux_to_sb(cube_or_img):
    """
    Wandelt Flusswerte (in FLUX_UNIT, pro Original-Pixel) in
    Oberflaechenhelligkeit (SB_UNIT, pro arcsec^2) um.
    Funktioniert sowohl fuer 2D-Bilder als auch 3D-Cubes.
    """
    return np.asarray(cube_or_img, dtype=float) / PIXEL_AREA_ARCSEC2

from astropy.modeling.models import Moffat2D, Gaussian2D

def make_source_psf(shape, fwhm_arcsec, pixel_scale=0.5, beta=3.):

    ny, nx = shape
    center_y, center_x = (ny - 1) / 2.0, (nx - 1) / 2.0
    fwhm_pix = fwhm_arcsec / pixel_scale
    
    y, x = np.mgrid[:ny, :nx]
    
    gamma = fwhm_pix / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))
    psf = Moffat2D(amplitude=1.0, x_0=center_x, y_0=center_y, gamma=gamma, alpha=beta)(x, y)
        
    return psf / np.nansum(psf)


def sb_to_flux(cube_or_img):
    """Kehrfunktion zu flux_to_sb (SB_UNIT -> FLUX_UNIT pro Original-Pixel)."""
    return np.asarray(cube_or_img, dtype=float) * PIXEL_AREA_ARCSEC2


with fits.open("/data/hetdex/u/bgrashey/cubes/ssa22_fullfp_stack.fits") as h:
    header = h[0].header
wcs = WCS(header)

COLNAMES = {
    "ra": ["ra", "RA", "Ra", "RAJ2000", "ra_vdfi", "ra_hetdex"],
    "dec": ["dec", "DEC", "Dec", "DEJ2000", "dec_vdfi", "dec_hetdex"],
    "z": ["z", "Z", "redshift", "REDSHIFT", "zspec", "ZSPEC", "z_vdfi", "z_hetdex", "redshift"],
    "flux": ["flux", "Flux", "FLUX", "flux_lya"],
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
        mask = dx ** 2 + dy ** 2 <= radius ** 2

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
    """
    Schneidet einen Sub-Cube um (ra, dec, z) aus. Rueckgabewerte sind in
    FLUX_UNIT (Fluss pro Original-Pixel), noch NICHT in SB umgerechnet.
    Gibt (None, None) zurueck, wenn die Quelle nicht vollstaendig im Cube liegt.
    """
    spec = (z + 1) * 1216  # Å
    x, y, z_pix = wcs.all_world2pix(ra, dec, spec, 0)
    xi, yi, zi = int(round(float(x))), int(round(float(y))), int(round(float(z_pix)))

    n_wave, ny, nx = zarr_cube.shape

    z0, z1 = zi - spec_width, zi + spec_width
    y0, y1 = yi - width, yi + width
    x0, x1 = xi - width, xi + width

    out_of_bounds = (z0 < 0 or z1 > n_wave or y0 < 0 or y1 > ny or x0 < 0 or x1 > nx)

    if out_of_bounds:
        return None, None  # Quelle liegt nicht vollstaendig im Cube

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
        center=(ny // 2, nx // 2),
        radius=ny // 4,
    )

    sky_subtracted = subtract_sky_sigmaclip(
        subcube,
        mask=mask,
        sigma=3.0,
        maxiters=5,
        stat="median",
    )

    return sky_subtracted


def subtract_continuum(cube, line_mask=None, degree=2, sigma=3.0, maxiters=5):
    n_wave, ny, nx = cube.shape
    wave = np.arange(n_wave)
    if line_mask is None:
        line_mask = np.zeros(n_wave, dtype=bool)
        center = n_wave // 2
        line_mask[center - 9:center + 10] = True

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
    """
    Diagnose-Hilfsfunktion: kollabiert einen (rohen, noch nicht gestackten)
    Sub-Cube entlang der Wellenlaengen-Achse zu einem 2D-Bild. Wird im
    eigentlichen Stacking-Pfad nicht mehr benutzt (siehe
    Stacking.narrowband_from_cube fuer das gestackte Cube), ist aber
    nuetzlich, um einzelne Quellen vor dem Stacking zu inspizieren.
    """
    n_wave, _, _ = subcube.shape
    wave = np.arange(n_wave)
    if line_mask is None:
        line_mask = np.zeros(n_wave, dtype=bool)
        center = n_wave // 2
        line_mask[center - 9:center + 10] = True

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


def scale_slice(img2d, wcs2d, target_wcs, npix):
    """
    Reprojiziert ein einzelnes 2D-Bild (z.B. eine Wellenlaengen-Scheibe
    eines Sub-Cubes, bereits in SB_UNIT) auf das Ziel-WCS. Gibt
    (regrid, footprint) zurueck.
    """
    regrid, footprint = reproject_exact(
        (img2d, wcs2d), target_wcs, shape_out=(npix, npix)
    )
    return regrid, footprint


class Stacking:
    """
    Klasse zum Stacken von Lyman-Alpha-Emittern. Benoetigt eine astropy
    Table und einen geladenen zarr-Cube.

    Workflow (stack()):
        1. pro Quelle: Sub-Cube ausschneiden (prepare_subcube)
        2. optional Sky-Subtraktion, dann Kontinuum-Subtraktion
        3. Umrechnung Fluss -> Oberflaechenhelligkeit (flux_to_sb)
        4. JEDE Wellenlaengen-Scheibe einzeln auf ein gemeinsames,
           physikalisches (kpc-)Gitter reprojizieren (scale_slice)
        5. footprint-gewichtete Mittelung ueber alle Quellen, Kanal fuer
           Kanal -> Ergebnis ist ein gestacktes CUBE, kein Bild.

    Das Ergebnis von stack() ist also (n_wave, npix, npix) in SB_UNIT
    (siehe Modul-Konstante SB_UNIT), wobei die spektrale Achse weiterhin
    pixelrelativ zur Linienmitte indiziert ist (0 = zentraler Kanal bei
    spec_width). Aus diesem Cube lassen sich nachtraeglich beliebige
    Schmalband-Bilder erzeugen, siehe narrowband_from_cube().

    Parameter
    ---------
    catalog : astropy Table
    cube : geladenes zarr-Array
    width : raeumliche Breite des Ausschnitts (in Pixeln, jede Seite)
    spec_width : spektrale Breite des Ausschnitts (in Pixeln, jede Seite)
    kpc_pxl : kpc pro Pixel im Zielraster
    npix : Kantenlaenge des gestackten Bildes/Cubes in Pixeln
    """

    def __init__(self, catalog, cube, width=25, spec_width=25, kpc_pxl=3, npix=50):
        self.catalog = catalog
        self.cube = cube
        self.width = width
        self.spec_width = spec_width
        self.kpc_pxl = kpc_pxl
        self.npix = npix
        self.n_wave = 2 * spec_width
        # relativer Pixel-Index zur Linienmitte, z.B. -25..24
        self.wave_pix = np.arange(self.n_wave) - spec_width
        self.stacked_cube = None  # wird von stack() befuellt
        self.stacked_psf = None

    def stack(self, do_sky_sub=False, do_cont_sub=False, normalize=False, verbose=True):
        """
        do_sky_sub : bool
            Wie im Original-Skript wird die Sky-Subtraktion standardmaessig
            NICHT angewendet (Oversubtraction-Problem). Auf True setzen,
            um sie zu aktivieren.
        """
        col_ra = _find_col(self.catalog, COLNAMES["ra"])
        col_dec = _find_col(self.catalog, COLNAMES["dec"])
        col_z = _find_col(self.catalog, COLNAMES["z"])

        cube_stack = []  # pro Quelle: (n_wave, npix, npix), SB_UNIT
        foot_stack = []  # pro Quelle: (n_wave, npix, npix)
        
        z_ref = np.median(self.catalog[col_z])

        n_skipped = 0

        for i in range(len(self.catalog)):
            ra = self.catalog[i][col_ra]
            dec = self.catalog[i][col_dec]
            z = self.catalog[i][col_z]
            
            col_lum = None
            if normalize:
                col_lum = _find_col(self.catalog, COLNAMES["luminosity"])

            subcube, sub_wcs = prepare_subcube(
                ra, dec, z, self.cube, width=self.width, spec_width=self.spec_width
            )

            if subcube is None:
                n_skipped += 1
                continue

            if normalize:
                L = float(self.catalog[i][col_lum])
 
                if not np.isfinite(L) or L <= 0:
                    n_skipped += 1
                    continue
 
            if do_sky_sub:
                subcube = subtract_sky_per_slice(subcube)
            
            if do_cont_sub:
                subcube = subtract_continuum(subcube)  # noch FLUX_UNIT
            
            sb_cube = flux_to_sb(subcube) * ((1.0 + z_ref) / (1.0 + z)) ** 3  # jetzt SB_UNIT
 
            if normalize:
                sb_cube = sb_cube / L

            target_wcs = make_wcs(ra, dec, z, kpc_per_pixel=self.kpc_pxl, npix=self.npix)

            regridded = np.full((self.n_wave, self.npix, self.npix), np.nan)
            footprint = np.zeros((self.n_wave, self.npix, self.npix))

            for k in range(self.n_wave):
                regrid_k, foot_k = scale_slice(sb_cube[k], sub_wcs, target_wcs, self.npix)
                regridded[k] = regrid_k
                footprint[k] = foot_k

            cube_stack.append(regridded)
            foot_stack.append(footprint)

        if len(cube_stack) == 0:
            raise RuntimeError("Keine gueltige Quelle zum Stacken uebrig (alles out of bounds?).")

        cube_stack = np.array(cube_stack)  # (n_sources, n_wave, npix, npix)
        foot_stack = np.array(foot_stack)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            valid = np.isfinite(cube_stack)
            foot_stack_masked = np.where(valid, foot_stack, 0.0)
            cube_filled = np.where(valid, cube_stack, 0.0)
            weighted_stack = np.nansum(cube_filled * foot_stack_masked, axis=0) / np.nansum(foot_stack_masked, axis=0)

        if verbose:
            print(f"Skipped: {n_skipped}")
            print(f"Stacked cube shape: {weighted_stack.shape}  (n_wave, npix, npix)")
            print(f"Stacked cube units: {SB_UNIT}")

        self.stacked_cube = weighted_stack
        return weighted_stack

    def narrowband_from_cube(self, half_width=15, mode="sum", stacked_cube=None):
        """
        Erzeugt ein Schmalband (NB)-Bild aus dem gestackten Cube, zentriert
        auf den mittleren Kanal (Linienmitte).

        half_width : Anzahl Kanaele links/rechts der Linienmitte, die
                      einbezogen werden (Gesamtbreite = 2*half_width + 1).
        mode :
            "mean" -> mittlere Oberflaechenhelligkeit ueber die gewaehlten
                      Kanaele. Einheit bleibt SB_UNIT.
            "sum"  -> Summe der SB ueber die Kanaele. Um daraus eine
                      integrierte SB in erg/s/cm^2/arcsec^2 zu bekommen,
                      muss extern noch mit der spektralen Dispersion
                      (AA/Pixel) multipliziert werden.
        """
        if stacked_cube is None:
            if self.stacked_cube is None:
                raise RuntimeError("Noch kein gestacktes Cube vorhanden - erst stack() aufrufen.")
            stacked_cube = self.stacked_cube

        n_wave = stacked_cube.shape[0]
        center = n_wave // 2
        sel = slice(max(center - half_width, 0), min(center + half_width + 1, n_wave))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if mode == "mean":
                nb = np.nanmean(stacked_cube[sel], axis=0)
            elif mode == "sum":
                nb = np.nansum(stacked_cube[sel], axis=0)
            elif mode == "median":
                nb = np.nanmedian(stacked_cube[sel], axis=0)
            else:
                raise ValueError("mode muss 'mean' oder 'sum' sein")

        return nb
    
    def stack_psf(self):
        
        col_ra = _find_col(self.catalog, COLNAMES["ra"])
        col_dec = _find_col(self.catalog, COLNAMES["dec"])
        col_z = _find_col(self.catalog, COLNAMES["z"])
        
        psf_stack = []
        foot_stack = []
        n_skipped = 0
        
        ny_sub, nx_sub = 2 * self.width, 2 * self.width
        
        for i in range(len(self.catalog)):
            ra = self.catalog[i][col_ra]
            dec = self.catalog[i][col_dec]
            z = self.catalog[i][col_z]
            
            subcube, sub_wcs = prepare_subcube(
                ra, dec, z, self.cube, width=self.width, spec_width=self.spec_width
            )
            if subcube is None:
                n_skipped += 1
                continue
            
            fwhm = 2.5
            if "psf" in self.catalog.colnames:
                val = self.catalog[i]["psf"]
                if np.isfinite(val) and val > 0:
                    fwhm = float(val)
            
            psf_raw = make_source_psf(
                shape=(ny_sub, nx_sub),
                fwhm_arcsec=fwhm,
            )
            
            target_wcs = make_wcs(ra, dec, z, kpc_per_pixel=self.kpc_pxl, npix=self.npix)
            regrid_psf, foot_psf = scale_slice(psf_raw, sub_wcs, target_wcs, self.npix)
            s = np.nansum(regrid_psf)
            if s>0:
                regrid_psf /= s
                
            psf_stack.append(regrid_psf)
            foot_stack.append(foot_psf)
            
            psf_stack = np.array(psf_stack)
            foot_stack = np.array(foot_stack)
            
            weighted_psf = np.nansum(psf_stack * foot_stack, axis=0) / np.nansum(foot_stack, axis=0)
            
            self.stacked_psf = weighted_psf / np.nansum(weighted_psf)
            
            return self.stacked_psf
                                                  

    def extract_sb_profile(
        self, img, error_map=None, center=None, r_max=40, dr_px=1.5, kpc_per_px=1.
    ):
        """Extrahiert ein 1D-Oberflaechenhelligkeitsprofil aus einem LAE-Stack.

        Parameters
        ----------
        img : 2D array
            Gestacktes Bild in SB_UNIT (z.B. erg/s/cm^2/arcsec^2).
        error_map : 2D array, optional
            Varianz- oder Standardabweichungs-Map des Stacks.
        center : tuple (x, y), optional
            Festes Zentrum (z.B. Bildmitte). Falls None, wird Bildmitte genutzt.
        r_max_px : float
            Maximaler Radius in Pixeln (sollte 40-60 kpc abdecken).
        dr_px : float
            Schrittweite der Ringe.
        """
        if center is None:
            ny, nx = img.shape
            center = ((nx - 1) / 2.0, (ny - 1) / 2.0)

        r_edges = np.arange(1.0, r_max + dr_px, dr_px)

        r_eff = []
        sb_profile = []
        sb_err = []

        for r_in, r_out in zip(r_edges[:-1], r_edges[1:]):
            annulus = CircularAnnulus(center, r_in=r_in, r_out=r_out)
            stats = ApertureStats(img, annulus, error=error_map)

            r_mid = np.sqrt(0.5 * (r_in**2 + r_out**2))
            r_eff.append(r_mid)

            sb_profile.append(stats.mean)

            if error_map is not None:
                sb_err.append(stats.mean_error)
            else:
                sb_err.append(stats.std / np.sqrt(stats.sum_aper_area.value))

        r_arcsec = np.array(r_eff) * kpc_per_px

        return (
            r_arcsec,
            np.array(sb_profile),
            np.array(sb_err),
        )
    
    
from photutils.segmentation import detect_sources
from photutils.background import MADStdBackgroundRMS
from scipy.ndimage import binary_dilation

def create_neighbor_mask(subcube, nsigma=3.0, npixels=4, target_protect_radius_pix=4, dilation_iters=2):
    """
    Erstellt eine 2D-Bool-Maske (True = maskieren / verwerfen).
    Maskiert helle Nachbarn, schuetzt aber das Bildzentrum (Zielquelle).
    """
    n_wave, ny, nx = subcube.shape
    
    # 1. 2D-Detektionsbild erzeugen (Median unterdrueckt schmale Linienemitte, zeigt Kontinuum)
    det_img = np.nanmedian(subcube, axis=0)
    
    # 2. Hintergrundrauschen robust bestimmen
    bkg_rms = MADStdBackgroundRMS().calc_background_rms(det_img)
    threshold = nsigma * bkg_rms
    
    # 3. Quellen detektieren (zusammenhaengende Pixel > threshold)
    segm = detect_sources(det_img, threshold=threshold, npixels=npixels)
    
    if segm is None:
        return np.zeros((ny, nx), dtype=bool)
    
    # 4. Maske aller Quellen holen (True = detektierte Quelle)
    source_mask = segm.data > 0
    
    # 5. Maske leicht vergroessern (PSF-Wings abdecken)
    if dilation_iters > 0:
        source_mask = binary_dilation(source_mask, iterations=dilation_iters)
        
    # 6. WICHTIG: Die Zielquelle im Zentrum schuetzen!
    yc, xc = (ny - 1) / 2.0, (nx - 1) / 2.0
    yy, xx = np.mgrid[:ny, :nx]
    dist_from_center = np.sqrt((xx - xc)**2 + (yy - yc)**2)
    
    # Zentrum freigeben: Zielquelle wird NICHT maskiert
    source_mask[dist_from_center <= target_protect_radius_pix] = False
    
    return source_mask    
    
    
def extract_all_subcubes(catalog, zarr_cube, width=25, spec_width=25):
    """
    Schneidet fuer jede gueltige Quelle im Katalog einen Subcube aus
    und sammelt sie zusammen mit WCS und Metadaten in einer Liste.
    """
    col_ra = _find_col(catalog, COLNAMES["ra"])
    col_dec = _find_col(catalog, COLNAMES["dec"])
    col_z = _find_col(catalog, COLNAMES["z"])
    has_lum = any(c in catalog.colnames for c in COLNAMES["luminosity"])
    col_lum = _find_col(catalog, COLNAMES["luminosity"]) if has_lum else None

    subcube_records = []
    n_skipped = 0

    for i in range(len(catalog)):
        row = catalog[i]
        ra = row[col_ra]
        dec = row[col_dec]
        z = row[col_z]
        lum = float(row[col_lum]) if col_lum else np.nan

        subcube, sub_wcs = prepare_subcube(
            ra, dec, z, zarr_cube, width=width, spec_width=spec_width
        )

        if subcube is None:
            n_skipped += 1
            continue

        subcube_records.append({
            "subcube": subcube,   # 3D numpy array
            "wcs": sub_wcs,       # 2D celestial WCS
            "ra": ra,
            "dec": dec,
            "z": z,
            "lum": lum,
        })

    print(f"Subcubes extrahiert: {len(subcube_records)} (Ausserhalb/Skipped: {n_skipped})")
    return subcube_records


def extract_all_subcubes_masked(catalog, zarr_cube, width=25, spec_width=25, mask_neighbors=True):
    col_ra = _find_col(catalog, COLNAMES["ra"])
    col_dec = _find_col(catalog, COLNAMES["dec"])
    col_z = _find_col(catalog, COLNAMES["z"])
    has_lum = any(c in catalog.colnames for c in COLNAMES["luminosity"])
    col_lum = _find_col(catalog, COLNAMES["luminosity"]) if has_lum else None

    subcube_records = []
    n_skipped = 0

    for i in range(len(catalog)):
        row = catalog[i]
        ra = row[col_ra]
        dec = row[col_dec]
        z = row[col_z]
        lum = float(row[col_lum]) if col_lum else np.nan

        subcube, sub_wcs = prepare_subcube(
            ra, dec, z, zarr_cube, width=width, spec_width=spec_width
        )

        if subcube is None:
            n_skipped += 1
            continue

        # --- Automatische Nachbar-Maskierung ---
        if mask_neighbors:
            mask = create_neighbor_mask(
                subcube, 
                nsigma=3.0, 
                npixels=4, 
                target_protect_radius_pix=4, # ca. 2.0" bei 0.5"/px
                dilation_iters=2
            )
            # Maskierte Spalten/Pixel ueber alle Kanaele auf NaN setzen:
            subcube = subcube.copy()
            subcube[:, mask] = np.nan

        subcube_records.append({
            "subcube": subcube,
            "wcs": sub_wcs,
            "ra": ra,
            "dec": dec,
            "z": z,
            "lum": lum,
        })

    print(f"Subcubes extrahiert: {len(subcube_records)} (Ausserhalb/Skipped: {n_skipped})")
    return subcube_records





import matplotlib.pyplot as plt
import numpy as np

class SourceMasker:
    def __init__(self, img2d, title="", default_radius=4.0):
        self.img = img2d
        self.default_radius = default_radius
        self.circles = []
        self.patches = []

        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        
        # Astronomische Skalierung
        vmin, vmax = np.nanpercentile(img2d, [5, 99])
        self.ax.imshow(img2d, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        self.ax.set_title(f"{title}\nLinks: Kreis | Rechts: Undo | 'q' / Schließen: Weiter")
        
        # Zielquelle in der Mitte markieren
        ny, nx = img2d.shape
        self.ax.plot((nx - 1) / 2.0, (ny - 1) / 2.0, "r+", markersize=14, markeredgewidth=2, label="Target")
        self.ax.legend(loc="upper right")

        self.cid_click = self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.cid_key = self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        if event.button == 1:  # Linksklick: Maske setzen
            xc, yc = event.xdata, event.ydata
            patch = plt.Circle((xc, yc), self.default_radius, color="red", alpha=0.45)
            self.ax.add_patch(patch)
            self.patches.append(patch)
            self.circles.append((xc, yc, self.default_radius))
            self.fig.canvas.draw()
        elif event.button == 3 and self.patches:  # Rechtsklick: Letzten Kreis löschen
            self.patches.pop().remove()
            self.circles.pop()
            self.fig.canvas.draw()

    def on_key(self, event):
        if event.key in ["q", "enter", "escape"]:
            plt.close(self.fig)

    def get_mask(self):
        ny, nx = self.img.shape
        mask = np.zeros((ny, nx), dtype=bool)
        yy, xx = np.mgrid[:ny, :nx]
        for xc, yc, r in self.circles:
            mask |= ((xx - xc) ** 2 + (yy - yc) ** 2) <= r ** 2
        return mask
    
    
    
    
    
def mask_subcubes_interactively(subcube_list, default_radius=4.0):
    """
    Geht die Liste von Subcubes der Reihe nach durch.
    Fuegt jedem Element in der Liste den Key 'mask' hinzu.
    """
    total = len(subcube_list)
    print(f"Starte Maskierung fuer {total} Subcubes...")
    print("Bedienung: Linksklick = Kreis setzen | Rechtsklick = Undo | 'q' = Naechste Quelle")

    for idx, item in enumerate(subcube_list):
        # Falls schon eine Maske existiert und du nicht ueberschreiben willst:
        if "mask" in item and item["mask"] is not None:
            continue

        cube = item["subcube"]
        # Kontinuums-/Nachbar-Detektionsbild via Median ueber Wellenlaenge
        det_img = np.nanmedian(cube, axis=0)

        title = f"Quelle {idx + 1}/{total} (z = {item['z']:.3f})"
        masker = SourceMasker(det_img, title=title, default_radius=default_radius)
        
        # Blockiert, bis das Fenster geschlossen wird
        plt.show(block=True)

        # Maske direkt im Dict abspeichern
        item["mask"] = masker.get_mask()
        
        n_masked_pixels = np.sum(item["mask"])
        print(f"[{idx + 1}/{total}] Gespeichert: {len(masker.circles)} Kreis(e), {n_masked_pixels} Pixel maskiert.")

    return subcube_list

import numpy as np
import warnings

class StackingFromSubcubes:
    """
    Stacking-Pipeline, die direkt mit einer Liste vorverarbeiteter,
    maskierter Subcubes gefüttert wird.
    Ergebnis ist wie im Original ein vollständiger 3D-Cube (n_wave, npix, npix).
    """
    def __init__(self, subcube_list, kpc_pxl=3, npix=50):
        if not subcube_list:
            raise ValueError("subcube_list darf nicht leer sein.")

        self.subcubes = subcube_list
        self.kpc_pxl = kpc_pxl
        self.npix = npix
        
        # Spektrale Dimensionen aus dem ersten Cube auslesen
        self.n_wave = self.subcubes[0]["subcube"].shape[0]
        self.spec_width = self.n_wave // 2
        self.wave_pix = np.arange(self.n_wave) - self.spec_width
        
        self.stacked_cube = None
        self.stacked_psf = None

    def stack(self, do_sky_sub=False, do_cont_sub=False, normalize=False, verbose=True):
        z_ref = np.median([item["z"] for item in self.subcubes])

        cube_stack = []  # (n_sources, n_wave, npix, npix)
        foot_stack = []  # (n_sources, n_wave, npix, npix)
        n_skipped = 0

        for idx, item in enumerate(self.subcubes):
            subcube = np.copy(item["subcube"])
            sub_wcs = item["wcs"]
            ra, dec, z = item["ra"], item["dec"], item["z"]
            lum = item.get("lum", np.nan)

            # Maskierte Pixel berücksichtigen, falls nicht schon im Array auf NaN gesetzt
            if "mask" in item and item["mask"] is not None and np.any(item["mask"]):
                subcube[:, item["mask"]] = np.nan

            if normalize:
                if not np.isfinite(lum) or lum <= 0:
                    n_skipped += 1
                    continue

            # Sky- und Kontinuum-Subtraktion wie im Original
            if do_sky_sub:
                subcube = subtract_sky_per_slice(subcube)

            if do_cont_sub:
                subcube = subtract_continuum(subcube)

            # Fluss -> Oberflächenhelligkeit + Cosmological Dimming
            sb_cube = flux_to_sb(subcube) * ((1.0 + z_ref) / (1.0 + z)) ** 3

            if normalize:
                sb_cube = sb_cube / lum

            # Ziel-WCS auf Basis der (ggf. rezentrierten) RA/Dec
            target_wcs = make_wcs(ra, dec, z, kpc_per_pixel=self.kpc_pxl, npix=self.npix)

            regridded = np.full((self.n_wave, self.npix, self.npix), np.nan)
            footprint = np.zeros((self.n_wave, self.npix, self.npix))

            # Jede Wellenlängenscheibe einzeln reprojizieren (wie im alten Skript)
            for k in range(self.n_wave):
                regrid_k, foot_k = scale_slice(sb_cube[k], sub_wcs, target_wcs, self.npix)
                regridded[k] = regrid_k
                footprint[k] = foot_k

            cube_stack.append(regridded)
            foot_stack.append(footprint)

        if len(cube_stack) == 0:
            raise RuntimeError("Keine gültigen Cubes zum Stacken übrig.")

        cube_stack = np.array(cube_stack)
        foot_stack = np.array(foot_stack)

        # Footprint-gewichtete Mittelung
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            valid = np.isfinite(cube_stack)
            foot_stack_masked = np.where(valid, foot_stack, 0.0)
            cube_filled = np.where(valid, cube_stack, 0.0)

            foot_sum = np.nansum(foot_stack_masked, axis=0)
            weighted_stack = np.where(
                foot_sum > 0,
                np.nansum(cube_filled * foot_stack_masked, axis=0) / foot_sum,
                np.nan
            )

        if verbose:
            print(f"Skipped: {n_skipped}")
            print(f"Stacked cube shape: {weighted_stack.shape}  (n_wave, npix, npix)")
            print(f"Stacked cube units: {SB_UNIT}")

        self.stacked_cube = weighted_stack
        return weighted_stack

    def narrowband_from_cube(self, half_width=15, mode="sum", stacked_cube=None):
        """Unverändert aus deinem Original-Code."""
        if stacked_cube is None:
            if self.stacked_cube is None:
                raise RuntimeError("Noch kein gestacktes Cube vorhanden - erst stack() aufrufen.")
            stacked_cube = self.stacked_cube

        n_wave = stacked_cube.shape[0]
        center = n_wave // 2
        sel = slice(max(center - half_width, 0), min(center + half_width + 1, n_wave))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if mode == "mean":
                nb = np.nanmean(stacked_cube[sel], axis=0)
            elif mode == "sum":
                nb = np.nansum(stacked_cube[sel], axis=0)
            elif mode == "median":
                nb = np.nanmedian(stacked_cube[sel], axis=0)
            else:
                raise ValueError("mode muss 'mean', 'sum' oder 'median' sein")

        return nb
    
    def extract_sb_profile(
        self, img, error_map=None, center=None, r_min=1, r_max=40, n_bins=10, kpc_per_px=1.
    ):
        """Extrahiert ein 1D-Oberflaechenhelligkeitsprofil aus einem LAE-Stack.

        Parameters
        ----------
        img : 2D array
            Gestacktes Bild in SB_UNIT (z.B. erg/s/cm^2/arcsec^2).
        error_map : 2D array, optional
            Varianz- oder Standardabweichungs-Map des Stacks.
        center : tuple (x, y), optional
            Festes Zentrum (z.B. Bildmitte). Falls None, wird Bildmitte genutzt.
        r_max_px : float
            Maximaler Radius in Pixeln (sollte 40-60 kpc abdecken).
        dr_px : float
            Schrittweite der Ringe.
        """
        if center is None:
            ny, nx = img.shape
            center = ((nx - 1) / 2.0, (ny - 1) / 2.0)

        r_edges = np.geomspace(r_min, r_max, n_bins+1)

        r_eff = []
        sb_profile = []
        sb_err = []

        for r_in, r_out in zip(r_edges[:-1], r_edges[1:]):
            annulus = CircularAnnulus(center, r_in=r_in, r_out=r_out)
            stats = ApertureStats(img, annulus, error=error_map)

            r_mid = np.sqrt(0.5 * (r_in**2 + r_out**2))
            r_eff.append(r_mid)

            sb_profile.append(stats.mean)

            if error_map is not None:
                sb_err.append(stats.mean_error)
            else:
                sb_err.append(stats.std / np.sqrt(stats.sum_aper_area.value))

        r_arcsec = np.array(r_eff) * kpc_per_px

        return (
            r_arcsec,
            np.array(sb_profile),
            np.array(sb_err),
        )
    
def load_subcubes_npz(filename="subcubes_for_laptop.npz"):
    data = np.load(filename, allow_pickle=True)
    n_items = int(data["n_items"])
    subcube_list = []

    for i in range(n_items):
        hdr = fits.Header.fromstring(str(data[f"wcs_hdr_{i}"]))
        item = {
            "subcube": data[f"cube_{i}"],
            "wcs": WCS(hdr),
            "ra": float(data[f"ra_{i}"]),
            "dec": float(data[f"dec_{i}"]),
            "z": float(data[f"z_{i}"]),
            "lum": float(data[f"lum_{i}"]),
            "mask": data[f"mask_{i}"] if f"mask_{i}" in data else None,
        }
        subcube_list.append(item)
        
    print(f"{len(subcube_list)} Subcubes erfolgreich geladen.")
    return subcube_list