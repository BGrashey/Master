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

    def stack(self, do_sky_sub=False, verbose=True):
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

        n_skipped = 0

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

            if do_sky_sub:
                subcube = subtract_sky_per_slice(subcube)

            contsub = subtract_continuum(subcube)  # noch FLUX_UNIT
            sb_cube = flux_to_sb(contsub)  # jetzt SB_UNIT

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
            weighted_stack = (
                np.nansum(cube_stack * foot_stack, axis=0)
                / np.nansum(foot_stack, axis=0)
            )

        if verbose:
            print(f"Skipped: {n_skipped}")
            print(f"Stacked cube shape: {weighted_stack.shape}  (n_wave, npix, npix)")
            print(f"Stacked cube units: {SB_UNIT}")

        self.stacked_cube = weighted_stack
        return weighted_stack

    def narrowband_from_cube(self, half_width=9, mode="mean", stacked_cube=None):
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
            else:
                raise ValueError("mode muss 'mean' oder 'sum' sein")

        return nb

    def cog(self, img, r_max=12):
        """
        Radiales Oberflaechenhelligkeits-Profil ("curve of growth").
        Erwartet ein 2D-Bild in SB_UNIT (z.B. aus narrowband_from_cube).

        Rueckgabe:
            radii_kpc   : Ring-Radien in kpc
            sb_profile  : MITTLERE Oberflaechenhelligkeit pro Ring (SB_UNIT)
            cum_sb_sum  : Summe der SB-Werte innerhalb der kumulativen
                          Apertur (kein echter Fluss - nur zur Diagnose
                          des Aufbaus einer Curve-of-Growth geeignet)
        """
        data_slice = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        x0, y0 = centroid_com(data_slice)

        radii = np.arange(2, r_max, 1)

        sb_profile = []
        cum_sb_sum = []

        for idx, r in enumerate(radii):
            if idx == 0:
                region = CircularAperture((x0, y0), r=r)
            else:
                region = CircularAnnulus((x0, y0), r_in=radii[idx - 1], r_out=r)

            stats = ApertureStats(data_slice, region)
            sb_profile.append(stats.mean)

            full_ap = CircularAperture((x0, y0), r=r)
            phot = aperture_photometry(data_slice, full_ap)
            cum_sb_sum.append(phot["aperture_sum"][0])

        radii_kpc = radii * self.kpc_pxl

        return radii_kpc, np.array(sb_profile), np.array(cum_sb_sum)