# Standard library
import os

# Core scientific stack
import numpy as np
from scipy.signal import convolve2d
from scipy.spatial import cKDTree

# Matplotlib
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

# Astropy
from astropy.convolution import Gaussian1DKernel, convolve_fft
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

# Regions / DS9
import pyregion
from regions import Regions

# Parallel / workflow
from dask import compute, delayed
from dask.distributed import get_client

# Utilities
from tqdm import tqdm

from .stack import create_3D_header



print("fof v0.5")

def get_virus_wlgrid():
    # returns generic VIRUS wavelength grid
    wlstep = 2
    wlstart = 3470. 
    wlgrid = np.arange( 1036 ) * wlstep + wlstart
    return wlgrid, wlstep, wlstart

def ztowl(z):
    wlgrid, wlstep, wlstart = get_virus_wlgrid()
    return z * wlstep + wlstart
    
def create_ds9_cmap():
    if 'ds9staircase' in plt.colormaps():
        return
    colors = []
    for ii in range(1,6):
        kk = ii/5.
        colors.append((kk*.3, kk*.3, kk*1))
    for ii in range(1,6):
        kk = ii/5.
        colors.append((kk*.3, kk*1,  kk*.3))
    for ii in range(1,6):
        kk = ii/5.
        colors.append((kk*1,  kk*.3, kk*.3))

    colors = np.array(colors)
    xx = np.linspace(0, 1, len(colors))

    newcolors = np.vstack([
        np.interp(xx, xx, colors[:,0]),
        np.interp(xx, xx, colors[:,1]),
        np.interp(xx, xx, colors[:,2]),
        np.ones_like(xx)
    ]).T

    ds9staircase = ListedColormap(newcolors, name='ds9staircase')
    matplotlib.colormaps.register(ds9staircase)   # register globally
    return ds9staircase

def to_2d_header(h):
    h = h.copy()
    kk = ["NAXIS3", "CTYPE3", "CRPIX3", "CDELT3", "CRVAL3", "CUNIT3"]
    for k in kk:
        if k in h:
            h.remove(k)
    return h

def mk_median_image(file_name, cen, win, out_file_name):
    """
    Compute a median out of a wavelength region of a cube and save to fits.
    """
    print("Opening fits cube ...")
    hdu = fits.open(file_name)
    print("Computing median ...")
    md = np.nanmedian(hdu[0].data[int(cen-win/2):int(cen+win/2)], axis=0)
    header = hdu[0].header.copy()
    hdu.close()
    
    print("Writing fits image ...")
    header = to_2d_header(header)
    imhdu = fits.PrimaryHDU(data=md, header=header)
    imhdu.header["history"] = f"Median collapsed around sclice {cen} with window {win} by Step030_fof.ipynb"
    imhdu.writeto(out_file_name, overwrite=True)
    return md, header

def mk_median_image_from_zarr(conf, cen, win, out_file_name):
    """
    Compute a median out of a wavelength region of a cube and save to fits.
    """
    print("Opening zarr cube ...")
    import zarr
    from vdfi import stack
    _ = conf.stack_sig_cube_name()
    fullcube = zarr.open_array(_, mode='r')
    md = np.nanmedian( fullcube[int(cen-win/2):int(cen+win/2)] , axis=0)
    header = stack.create_3D_header(conf)
    
    print("Writing fits image ...")
    header = to_2d_header(header)
    imhdu = fits.PrimaryHDU(data=md, header=header)
    imhdu.header["history"] = f"Median collapsed around sclice {cen} with window {win} by Step031_fof.ipynb"
    imhdu.writeto(out_file_name, overwrite=True)
    return md, header

def visualize_image(file_name, ds9regions, vmin=-0.0, vmax=1., ax=None):
    """
    Show 2D fits image, optionally overplot ds9 region file(s).
    """
    from astropy.wcs import wcs

    header = fits.getheader(file_name).copy()
    md     = fits.getdata(file_name).copy()
    w = wcs.WCS(header)
    
    if ax == None:
        f = plt.figure(figsize=[12,12])
        ax = plt.subplot(111, projection = w)
    
    plt.imshow( md, vmin=vmin, vmax=vmax, origin='lower')
    
    for ds9region in ds9regions:
            print(f"ds9region: {ds9region}")
            with open(ds9region, 'r') as fin:
                region_string = fin.read()
    
            print(f"parsing regions")
            r2 = pyregion.parse(region_string).as_imagecoord(header)
            print(f"converting regions to patches")
            patch_list, artist_list = r2.get_mpl_patches_texts()
    
            # Hack to deal with the fact that pyregions does not properly set the 
            # fill attribute
            for s,p in zip(r2,patch_list):
                p.fill = False
                if "fill" in s.attr[1]:
                        p.fill = (s.attr[1]["fill"]) == '1 '

            # ax is a mpl Axes object
            for j, p in enumerate(patch_list):
                if p.fill:
                    p.set_facecolor(p.get_edgecolor())
                p.set_fill(True)
                p.set_facecolor('#AAAAAA')
                p.set_edgecolor('#AAAAAA')
                p.set_linewidth( p.get_linewidth() * 1. )
                p.set_linestyle("-")
    
                ax.add_patch(p)


def make_pixelated_mask(sig_file_name, ds9regions, mask_file_name, overwrite=True):
    """
    Takes single (consolidated) ds9 regions file and a fit cube (or image)
    and generates a pixelated mask.
    """
    
    if not os.path.exists(mask_file_name) or overwrite:
        # this takes a while
        regions_file = ds9regions
    
        print(f"Loading {regions_file} ...")
        
        # Warning, this takes a while
        r = pyregion.open(regions_file)

        header = to_2d_header( fits.getheader(sig_file_name) )
        w = WCS( header )

        print(f"Computing mask  ...")
        mymask = r.get_mask(header=header, shape=[header['NAXIS2'], header['NAXIS1']])

        print(f"Writing mask file {mask_file_name} ...")
        mask_hdu = fits.PrimaryHDU(data=np.array(mymask, dtype=int), header=header )
        mask_hdu.writeto(mask_file_name, overwrite=True )
    else:
        print(f"Loading already existing {mask_file_name} ...")
        mask_hdu = fits.open(mask_file_name)
        mymask = mask_hdu[0].data.copy()
        mask_hdu.close()

    return np.array(mymask, dtype=bool) # save some memory by saving it as bool


def resample_aux_image(target_grid_fits_file, aux_fits_file, resampled_aux_file, target_header=None, ext=1, FORCE_REDO=True ):
    """
    Resample auxiallary image (e.g. deep ground based data. to the same pixel grid 
    as we are using in this work. """
    
    if not os.path.exists(resampled_aux_file) or FORCE_REDO:
        from astropy.io import fits
        from astropy.wcs import WCS
        from reproject import reproject_interp  # or: from reproject import reproject_exact
        
        # --- read target image (defines output grid) ---

        if target_grid_fits_file == None and target_header == None:
            print("Error: target_grid_fits_file and target_header connot both be None.")

        if target_header == None:
            target_header = fits.getheader(target_grid_fits_file).copy()
        wcs1  = WCS(target_header)
        wcs1 = wcs1.dropaxis(2)
        
        # --- read source image (to be resampled) ---
        hdu2 = fits.open(aux_fits_file)[ext]
        data2 = hdu2.data
        wcs2  = WCS(hdu2.header)
        
        # --- reproject source onto target grid ---
        reproj_func = reproject_interp        # fast, not flux-conserving

        shape = [target_header['NAXIS2'], target_header['NAXIS1']]
        data2_on_1, footprint = reproj_func((data2, wcs2), wcs1, shape_out=shape)
        
        # --- write result with target WCS ---
        print(f"Writing to {resampled_aux_file} ...")
        fits.writeto(resampled_aux_file, data2_on_1, to_2d_header(target_header), overwrite=True)

    else:
        data2_on_1 = fits.getdata(resampl_hsc)

    return data2_on_1


def circular_kernel(radius):
    r = int(np.ceil(radius))
    size = 2 * r + 1  # ensures center + radius fits
    y, x = np.ogrid[-r:r+1, -r:r+1]
    mask = x**2 + y**2 <= radius**2
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[mask] = 1.0
    return kernel


def grow_mask(mymask, radius):
    # grow mask by convolution to get rid of halos and edge glow
    
    # Define convolution kernel (3x3 ones -> grow by 1 pixel in all directions)
    kernel = circular_kernel(radius)
    
    # Convolve
    conv = convolve2d(mymask, kernel, mode="same", boundary="fill", fillvalue=0)
    
    grown_mymask = (conv > 0).astype(bool)

    return grown_mymask


def regions_global_bbox_and_mask(sig_file_name, ds9_region_path, hdu_index=0):
    """
    Returns (xmin, ymin, xmax, ymax, mask) where:
      - xmin, ymin, xmax, ymax: pixel coordinates (0-based, inclusive)
        of the union bounding box of all regions, clipped to the image extent.
      - mask: 2D boolean numpy array, same shape as the image,
        True for pixels inside any region.
    
    Parameters
    ----------
    hdul : astropy.io.fits.HDUList
        Open FITS HDU list.
    ds9_region_path : str
        Path to DS9 region file.
    hdu_index : int, optional
        Which HDU to use (default=0).
    """
    # Select image + WCS

    hdul = fits.open(sig_file_name)
    data = hdul[hdu_index].data
    header = hdul[hdu_index].header.copy()
    ny, nx = data.shape[-2:]

    wcs = WCS(header)
    wcs = wcs.dropaxis(2)

    # Read DS9 regions
    regs = Regions.read(ds9_region_path, format="ds9")

    xmin = np.inf
    ymin = np.inf
    xmax = -np.inf
    ymax = -np.inf
    mask = np.zeros((ny, nx), dtype=bool)

    for reg in regs:
        # Convert to pixel region if needed
        if hasattr(reg, "to_pixel"):
            reg = reg.to_pixel(wcs)

        # Update bounding box
        bb = reg.bounding_box
        xmin = min(xmin, int(np.floor(bb.ixmin)))
        xmax = max(xmax, int(np.ceil(bb.ixmax)))
        ymin = min(ymin, int(np.floor(bb.iymin)))
        ymax = max(ymax, int(np.ceil(bb.iymax)))

        # Build region mask and paste into global mask
        rmask = reg.to_mask(mode="center")
        submask = rmask.to_image((ny, nx))
        if submask is not None:
            mask |= submask.astype(bool)

    if not np.isfinite([xmin, ymin, xmax, ymax]).all():
        raise ValueError("No regions produced a valid bounding box.")

    # Clip bounding box to image extent
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(nx - 1, xmax)
    ymax = min(ny - 1, ymax)

    hdul.close()
    return xmin, ymin, xmax, ymax, mask



def gaussian3d_kernel(sigmas, truncate=4.0, normalize=True, dtype=np.float64):
    """
    Axis-aligned 3D Gaussian kernel with different sigmas per axis.

    Parameters
    ----------
    sigmas : tuple(float)  # (σz, σy, σx) in pixels
    truncate : float       # radius = truncate*σ on each axis
    normalize : bool       # if True, kernel sums to 1
    dtype : np.dtype

    Returns
    -------
    g : ndarray, shape ((2*rz+1), (2*ry+1), (2*rx+1))
    """
    sz, sy, sx = (float(s) for s in sigmas)
    rz, ry, rx = (int(truncate*s + 0.5) for s in (sz, sy, sx))

    z = np.arange(-rz, rz+1, dtype=dtype)[:, None, None]
    y = np.arange(-ry, ry+1, dtype=dtype)[None, :, None]
    x = np.arange(-rx, rx+1, dtype=dtype)[None, None, :]

    gz = np.exp(-0.5*(z/sz)**2)
    gy = np.exp(-0.5*(y/sy)**2)
    gx = np.exp(-0.5*(x/sx)**2)

    g = gz * gy * gx  # separable outer product
    if normalize:
        g /= g.sum()
    return g


def fill_master_dict(conf, fields):

    # generic VIRUS wavelength grid
    wlgrid, wlstep, wlstart = get_virus_wlgrid()


    stack_fits_name = conf.stack_fits_name()
    print(f"flux cube fits name:     {stack_fits_name} ...")
    contsub_fits_file = conf.stack_contsub_fits_name()
    print(f"contsub flux cube fits name: {contsub_fits_file} ...")
    sig_file_name =conf.stack_sig_fits_name()
    print(f"significance cube name:     {sig_file_name} ...")
    stack_filtered_sig_fits_name = conf.stack_filtered_sig_fits_name()
    print(f"filtered significance cube name: {stack_filtered_sig_fits_name} ...")

    
    flux_hdu = fits.open(stack_fits_name)
    flux_cube = flux_hdu[0].data
    errflux_cube = flux_hdu[1].data
    
    # contsub data are needed for fluxing
    contsub_hdu = fits.open(contsub_fits_file)
    contsub_flux_cube = contsub_hdu[0].data

    sig_hdu = fits.open(sig_file_name)
    sig_cube = sig_hdu[0].data
    
    filtered_sig_hdu  = fits.open(stack_filtered_sig_fits_name)
    filtered_sig_cube = filtered_sig_hdu[0].data
    
    print("Creating master dict ...")
    for name in fields:
        print(f"  {name}:")
    
        print("   Loading field region ...")
        # This reads ds9 region files which define the (sub) regions of interest
        # and computed the extent of the regions in x and y
        fields[name]["xstart"], fields[name]["ystart"], fields[name]["xend"], fields[name]["yend"], fields[name]["field_mask"] =\
            regions_global_bbox_and_mask(sig_file_name, fields[name]["field_reg_file"])
    
        xstart,ystart,xend,yend,field_mask=\
            fields[name]["xstart"], fields[name]["ystart"], fields[name]["xend"], fields[name]["yend"], fields[name]["field_mask"] 

        sl_low = fields[name]["sl_low"]
        sl_hi  = fields[name]["sl_hi"] 
       
        print("   ", f"Spatal footprint:")
        print("   ", f" xstart, xend = {xstart}, {xend}")
        print("   ", f" ystart, yend = {ystart}, {yend}")

        wlcen = int((sl_hi + sl_low)/2.)
        print("   ", f"Central slice {wlcen}")
        print("   ", f"Central wavelength {wlgrid[wlcen]}")
        print("   ", f"Start slice {sl_low}")
        print("   ", f"End of slice {sl_hi}")
        print("   ", f"Number of slices {sl_hi-sl_low}")
        print("   ", f"Wavelength range {wlgrid[sl_low]-wlstep/2.} A - {wlgrid[sl_hi]+wlstep/2.} A. ")

        print("   ", "Filling flux cube ...")
        fields[name]["flux"]         = flux_cube[sl_low:sl_hi, ystart:yend, xstart:xend].copy() # on band range from paper
        print("   ", "Filling flux error cube ...")
        fields[name]["errflux"]      = errflux_cube[sl_low:sl_hi, ystart:yend, xstart:xend].copy() # on band range from paper
        print("   ", "Filling flux contsub cube ...")
        fields[name]["contsub"]      = contsub_flux_cube[sl_low:sl_hi, ystart:yend, xstart:xend].copy() # on band range from paper
        print("   ", "Filling flux SN cube ...")
        fields[name]["sn"]           = sig_cube[sl_low:sl_hi, ystart:yend, xstart:xend].copy() # on band range from paper
        print("   ", "Filling filtered SN cube ...")
        fields[name]["filtered_sn"]  = filtered_sig_cube[sl_low:sl_hi, ystart:yend, xstart:xend].copy() # on band range from paper
    
        zz,yy,xx = [np.arange(s, dtype=int) for s in fields[name]["sn"].shape]
        
        fields[name]["zz"] = zz
        fields[name]["yy"] = yy
        fields[name]["xx"] = xx

        print("    Generating subcube wcs ...")
        # generate appropriate header for subcube
        header = sig_hdu[0].header.copy() 
        
        header["CRPIX1"] = header["CRPIX1"]-fields[name]["xstart"]
        header["CRPIX2"] = header["CRPIX2"]-fields[name]["ystart"]
        header["CRPIX3"] = header["CRPIX3"]-sl_low
        
        subw = WCS(header)
        
        fields[name]["subw"]   = subw     
        fields[name]["header"] = header     
        fields[name]["wlgrid"] = wlgrid[sl_low:sl_hi]
    
        print("   ")
    
    # clear memory
    flux_cube = None     
    sig_cube = None     
    contsub_flux_cube = None
    filtered_sig_cube = None
    
    print('done')


def apply_spatial_mask(grown_spatial_mask, fields):
    from tqdm.notebook import tqdm
    print("Applying spatial masks ...")
    
    for name in fields:
        print("   ", name)
        field_mask       = fields[name]["field_mask"] 
        xstart,ystart,xend,yend,field_mask=\
            fields[name]["xstart"], fields[name]["ystart"], fields[name]["xend"], fields[name]["yend"], fields[name]["field_mask"] 

        _grown_mymask    = grown_spatial_mask + ~field_mask
        sub_grown_mymask = _grown_mymask[ystart:yend, xstart:xend]
        
        fields[name]["sub_grown_mymask"] = sub_grown_mymask

        for k in tqdm(["sn", "filtered_sn", "flux", "errflux", "contsub"]):
            subcube = fields[name][k]
            for i in range( 0, subcube.shape[0]):
                subcube[i][ sub_grown_mymask ] = np.nan
                subcube[i][ subcube[i] == 0. ] = np.nan # also eliminate zeros.

    print('Done.')

    

def apply_wavelength_mask(bad_wl_ranges, fields):
    print("Applying wavelength mask ...")

    wlgrid,_,_ = get_virus_wlgrid()
    
    wlmask = np.zeros_like( wlgrid, dtype=bool )
    for bwlr in bad_wl_ranges:
        wlmask += (wlgrid > bwlr[0]) * (wlgrid < bwlr[1]) 
        
    for name in fields:
        print("   ", name)
        sl_low  = fields[name]["sl_low"] 
        sl_hi   = fields[name]["sl_hi"] 
        
        fields[name]["zmask"] = wlmask[sl_low:sl_hi] # store for later use (e.g. plotting)
        print(f"        Setting {np.sum(wlmask[sl_low:sl_hi])} slizes to nan")
        for k in ["sn", "filtered_sn"]:
            fields[name][k][ fields[name]["zmask"]  ] = np.nan

    print("Done.")


def create_subfield_plots(bad_wl_ranges, fields):
    for name in fields:
        print("   ", name)
        fig = plt.figure(figsize=[15,5])
        ax1 = plt.subplot(131)
    
        for bwlr in bad_wl_ranges:
            ax1.axvspan(bwlr[0], bwlr[1], color='lightgrey', alpha=0.4, hatch='//')
            
        plt.text(.5,.99, "x/y averaged SN: " + name, transform=ax1.transAxes, ha='center', va='top' )    
        plt.plot(fields[name]["wlgrid"], np.nanmean(np.nanmean(fields[name]["sn"], axis=2), axis=1))
        plt.xlabel("wavelength")
    
        ax2 = plt.subplot(132)
        plt.text(.5,.99, "Z-Integrated SN: " + name, transform=ax2.transAxes, ha='center', va='top', color='white' )    
        plt.imshow(np.nansum(fields[name]["sn"], axis=0), origin='lower',vmin=0,vmax=1)
    
        ax3 = plt.subplot(133)
        plt.text(.5,.99, "Mask: " + name, transform=ax3.transAxes, ha='center', va='top' , color='white')    
        plt.imshow( fields[name]["sub_grown_mymask"], origin='lower')
        
        plt.show()


def filter_subcubes(fields, sigma_spatial, sigma_spectral, truncate, FORCE_REDO=True):
    
    K = gaussian3d_kernel(sigmas=(sigma_spectral, sigma_spatial, sigma_spatial), truncate=truncate) 
    
    print("Filtering SN subcubes spectrally and spatially ...")
    for name in fields:
        print("   ", name)
        if "filtered_sn" in fields[name] and not FORCE_REDO:
            print(f"Field {name} already has a fitered version, please set FORCE_REDO=True to overwrite...")
            continue
        
        # convolve_fft can take a tuple of 1D kernels via separable=True
        fsubcube = convolve_fft(
            fields[name]["sn"], K,
            allow_huge=True,       # if the cube is large
            nan_treatment='fill',
            boundary='fill', fill_value=0.,  # or 'extend'/'wrap' per your use case
        )
        
        fsubcube[np.isnan(fields[name]["sn"])] = np.nan
        fields[name]["filtered_sn"] = fsubcube


def create_filtered_subfield_plots(fields, cmap):
    print("Displaying filtered subcubes  ...")
    for name in fields:
        print("   ", name)
    
        fig= plt.figure(figsize=[40,30])
        #plt.title(name)
        ax = plt.subplot(121)

        #sl_hi = fields[name]["sl_hi"]
        #sl_low = fields[name]["sl_low"]
        #cen = (sl_hi-sl_low)//2
        cen = fields[name]["sn"].shape[0]//2

        image = plt.imshow(fields[name]["sn"][cen], vmin =-1,  vmax = 2., origin='lower', cmap=cmap)
        
        cbar = fig.colorbar(image, ax=ax, pad=0.01, fraction=0.046)
        cbar.set_label("SN")
        
        
        ax = plt.subplot(122)
        image = plt.imshow(fields[name]["filtered_sn"][cen], vmin =-5,  vmax = 8.5, origin='lower', cmap=cmap)
        
        cbar = fig.colorbar(image, ax=ax, pad=0.01, fraction=0.046)
        cbar.set_label("SN")
        
        fig.tight_layout()
        plt.show()

def threshold_pixels(threshold, fields, apply_on_filtered):
    print(f"Thresholding spaxels with threshold = {threshold}.")
    for name in fields:
        print("   ", name)
        #ii = fields[name]["sn"] > threshold
        if apply_on_filtered:
            ii = fields[name]["filtered_sn"] > threshold
        else:
            ii = fields[name]["sn"] > threshold
        fields[name]["thresheld"] = ii
        
        #ii = subcube > 0.00065
        N = np.sum(ii)
        print("   ", f"{N} spaxels fall above SN threshold of {threshold}")
        fields[name]["selected_spaxel_indices"] = ii
        YY,ZZ,XX = np.meshgrid(fields[name]["yy"], fields[name]["zz"], fields[name]["xx"])
        x = XX[ii] 
        y = YY[ii] 
        z = ZZ[ii] 
        sn      = fields[name]["sn"][ii]
        fsn     = fields[name]["filtered_sn"][ii]
        contsub = fields[name]["contsub"][ii]
        
        fields[name]["selected_spaxels"] = {'x' : x, 'y' : y, 'z' : z, 'sn' : sn, 'fsn' : fsn, 'contsub' : contsub}

def friends_of_friends_3d_nonrecursive(positions, sn, cont_sub, linking_length, workers):
    """
    Perform a 3D Friends-of-Friends (FoF) clustering algorithm on a set of particle positions.
    Non-recursive version using an explicit stack for breadth-first search (BFS).

    Parameters:
    - positions: An Nx3 numpy array or list of tuples containing 3D positions.
    - linking_length: The maximum linking length for the FoF algorithm.

    Returns:
    - clusters: A list of lists, where each inner list contains indices of particles in a cluster.
    """
    # Build KD-tree for neighbor searches
    print("     ", "KDTree ...")
    kdtree = cKDTree(positions)

    num_particles = len(positions)
    particle_visited = np.zeros(num_particles, dtype=bool)
    clusters = []
    clusters_sn = []
    clusters_cont_sub = []

    print("     ", f"Loop through all {num_particles} particles...")
    for particle_index in tqdm(range(num_particles)):
        if particle_visited[particle_index]:
            continue

        # Start new cluster
        cluster          = []
        cluster_sn       = []
        cluster_cont_sub = []
        queue = [particle_index]
        particle_visited[particle_index] = True

        while queue:
            current = queue.pop()
            cluster.append(current)
            cluster_sn.append(sn[current])
            cluster_cont_sub.append(cont_sub[current])
            # Find neighbors within linking length
            neighbors = kdtree.query_ball_point(positions[current], r=linking_length, workers=workers)
            for neighbor in neighbors:
                if not particle_visited[neighbor]:
                    particle_visited[neighbor] = True
                    queue.append(neighbor)

        clusters.append(cluster)
        clusters_sn.append(cluster_sn)
        clusters_cont_sub.append(cluster_cont_sub)
        n = len(clusters)

    return clusters, clusters_sn, clusters_cont_sub


def runfof(x, y, z, sn, cont_sub, linking_length, workers=18):
    print("     ", "Generate list of positions ...")
    positions = list(zip(x[:], y[:], z[:]))

    print("     ", "Launch FoF ...")
    clusters, cluster_values, clusters_cont_sub = \
        friends_of_friends_3d_nonrecursive(positions, sn, cont_sub, linking_length, workers)

    print(f"      Number of clusters: {len(clusters)}")
    return clusters, cluster_values, clusters_cont_sub


def execute_fof_on_fields(fields, linking_length):
    print("Running FoF ...")
    for name in fields:
        print("   ", name)
       
        #############
        print("Applying bad wavelength region mask ...")
        import numpy as np
        from scipy.interpolate import interp1d
        
        
        # Cubic interpolator: wavelength -> pixel index
        field_wlgrid = fields[name]['wlgrid']
        wl_zindices  = np.arange(len(field_wlgrid))
        #print( field_wlgrid, wl_zindices )
        wl_to_pix = interp1d(
            field_wlgrid, wl_zindices,
            kind="cubic",
            bounds_error=False,     # don't crash outside range
            fill_value=np.nan,      # or "extrapolate" if you prefer
            assume_sorted=False     # set True if wl is strictly sorted already
        )
        print("Subcube wlsclices run from ", field_wlgrid[0],"A to",  field_wlgrid[-1], "A")
        
        z = fields[name]["selected_spaxels"]['z']
        
        mask     =  np.array(np.ones_like(z), dtype=bool)

        bad_wl_ranges = fields[name]["bad_wl_ranges"]
        for i, bwlr in enumerate(bad_wl_ranges):
            print(" Bad wavelength region ", bwlr[0], "A ", bwlr[1])
            if  bwlr[1] < field_wlgrid[0] or bwlr[0] > field_wlgrid[-1]:
                print("   does not cover subcube wl slices.")
                continue
            if bwlr[0] < field_wlgrid[0]:
                bwlr[0] = field_wlgrid[0]
            if bwlr[1] > field_wlgrid[-1]:
                bwlr[1] = field_wlgrid[-1]
            zstart, zend = wl_to_pix([bwlr[0], bwlr[1]])  # pixel indices (float)
            zstart = int(np.round(zstart))
            zend  = int(np.round(zend))
            print("    ", bwlr[0], "A ", bwlr[1], "A -> slices", zstart," - ",zend )
            _ = (z >= zstart) *  (z <= zend)
            print("    masking: ", np.sum(_))
            mask = mask * ~ _
            
        
        print("Finally masking ", np.sum(~mask), ' flagged (from thresholding) spaxels.') 

        x        = fields[name]["selected_spaxels"]['x'][mask]
        y        = fields[name]["selected_spaxels"]['y'][mask]
        z        = fields[name]["selected_spaxels"]['z'][mask]
        sn       = fields[name]["selected_spaxels"]['sn'][mask]
        cont_sub = fields[name]["selected_spaxels"]['contsub'][mask]
        fields[name]["spaxel_mask"] = mask
        #############
        
        # FOF EXECUTION
        fields[name]["clusters"], fields[name]["cluster_sn"], fields[name]["cluster_contsub"] = \
            runfof(x, y, z, sn, cont_sub, linking_length = linking_length ) # sqrt(3) shoudl be the approriate number in 3D to allow for cubes
    
        # find largest cluster
        cluster_sizes = [len(c) for c in fields[name]["clusters"]]
        maxindex = np.argmax(cluster_sizes)
        N = len(fields[name]["clusters"][maxindex])
        print(f"      Largest group is {maxindex} with {N} members")
        
        # revers sorted list of indices of largest clusters (largest first)
        sizeindices = np.argsort(cluster_sizes)[::-1]
    
        fields[name]["cluster_sizes"] = cluster_sizes
        fields[name]["sizeindices"] = sizeindices
        print("   ")


def compute_segmentation_maps(fields, segment_threshold, thresheld_fraction=.05):
    print(f"Only maintaining segments that contain spaxels with a value >= {segment_threshold} at a fraction of at least {thresheld_fraction} of all spaxels in the cluster ")
    for name in fields:
        print("   ", name)
        YY,ZZ,XX = np.meshgrid(fields[name]["yy"], fields[name]["zz"], fields[name]["xx"])
    
        sn = fields[name]["sn"]
        clusters = fields[name]["clusters"]
    
        segmap  = np.zeros_like(sn) - 1. # otherwise first object is hidden
        sizemap = np.zeros_like(sn)
        for i,c in tqdm(list(enumerate(clusters))):
            x = fields[name]["selected_spaxels"]['x']
            y = fields[name]["selected_spaxels"]['y']
            z = fields[name]["selected_spaxels"]['z']
            _ = sn[z[c],y[c],x[c]] >= segment_threshold
            if (_).any() and (np.sum(_)/len(c)) > thresheld_fraction:
                
                    segmap[z[c],y[c],x[c]] = i
                    sizemap[z[c],y[c],x[c]] = len(c)
        
            
        fields[name]["segmap"] = segmap
        fields[name]["sizemap"] = sizemap



def build_catalogs(fields, nmax=None):
    names = ["id", "xwm", "ymw", "zwm", "rawm","decwm","wlwm","nspaxels","summed_SN",\
             "max_SN","summed_flux","max_flux","xextent","yextent","zextent","xsigma","ysigma","zsigma" ]
    dtype = [int,float,float,float,float,float,float,int,float,float,float,float,float,float,float,float,float,float]
    
    
    for name in list(fields.keys()):
        rows = []
        print(name)
        sl_low = fields[name]["sl_low"]
        sl_hi = fields[name]["sl_hi"]
        x = fields[name]["selected_spaxels"]['x']
        y = fields[name]["selected_spaxels"]['y']
        z = fields[name]["selected_spaxels"]['z']
        clusters = fields[name]["clusters"]
        cluster_sn = fields[name]["cluster_sn"]
        cluster_contsub = fields[name]["cluster_contsub"]
    
        patches = []
        print(f"N = {len(clusters)}")
        if nmax == None:
            nmax = len(clusters)
        for id,(c, cv, cs) in tqdm( list( enumerate(zip(clusters, cluster_sn, cluster_contsub)))[:nmax] ):
            nspaxels = len(c)
    
            summed_SN = np.nansum(cv)
            max_SN = np.nanmax(cv)
    
            summed_flux = np.nansum(cs) * 2 # BACAUSE IT IS 2 A PER SPAXEL
            max_flux = np.nanmax(cs)
            
            xwm = np.nansum(x[c] * cv) / summed_SN
            ywm = np.nansum(y[c] * cv) / summed_SN
            zwm = np.nansum(z[c] * cv) / summed_SN
    
            xsigma = np.nansum((x[c] - xwm)**2. * cv) / summed_SN
            ysigma = np.nansum((y[c] - ywm)**2. * cv) / summed_SN
            zsigma = np.nansum((z[c] - zwm)**2. * cv) / summed_SN
            wlwm = ztowl(zwm + sl_low)
            sc = fields[name]["subw"].dropaxis(2).pixel_to_world(xwm,ywm)
            rawm = float(sc.ra.deg)
            decwm = float(sc.dec.deg)
            
            xextent = np.max(x[c]) - np.min(x[c])
            yextent = np.max(y[c]) - np.min(y[c])
            zextent = np.max(z[c]) - np.min(z[c])
    
            r = [id, xwm, ywm, zwm, rawm, decwm, wlwm, nspaxels, \
                 summed_SN, max_SN, summed_flux,max_flux, xextent, yextent, zextent, xsigma, ysigma, zsigma] 
            #telcatalog.add_row(r)
            rows.append(r)
    
        telcatalog = Table(names=names, dtype=dtype, rows=rows)
    
        #telcatalog = vstack(tables)
    
        fields[name]["telcatalog"] = telcatalog


import numpy as np
from astropy.table import Table
from dask import delayed
from dask.distributed import get_client, as_completed
from tqdm.notebook import tqdm


def build_catalogs_dask(fields, nmax=None):
    names = [
        "id", "xwm", "ymw", "zwm", "rawm", "decwm", "wlwm",
        "nspaxels", "summed_SN", "max_SN",
        "summed_flux", "max_flux",
        "xextent", "yextent", "zextent",
        "xsigma", "ysigma", "zsigma"
    ]
    dtype = [
        int, float, float, float, float, float, float,
        int, float, float,
        float, float,
        float, float, float,
        float, float, float
    ]

    client = get_client()  # REQUIRED for progress tracking

    @delayed
    def process_one_cluster(
        cid, c, cv, cs,
        x, y, z,
        sl_low, subw
    ):
        summed_SN = np.nansum(cv)
        xwm = np.nansum(x[c] * cv) / summed_SN
        ywm = np.nansum(y[c] * cv) / summed_SN
        zwm = np.nansum(z[c] * cv) / summed_SN

        xsigma = np.nansum((x[c] - xwm)**2 * cv) / summed_SN
        ysigma = np.nansum((y[c] - ywm)**2 * cv) / summed_SN
        zsigma = np.nansum((z[c] - zwm)**2 * cv) / summed_SN

        wlwm = ztowl(zwm + sl_low)

        sc = subw.dropaxis(2).pixel_to_world(xwm, ywm)
        rawm  = float(sc.ra.deg)
        decwm = float(sc.dec.deg)

        return [
            cid, xwm, ywm, zwm, rawm, decwm, wlwm, len(c),
            np.nansum(cv), np.nanmax(cv),
            np.nansum(cs) * 2.0, np.nanmax(cs),
            np.max(x[c]) - np.min(x[c]),
            np.max(y[c]) - np.min(y[c]),
            np.max(z[c]) - np.min(z[c]),
            xsigma, ysigma, zsigma
        ]

    for name in fields:
        print(name)

        f = fields[name]
        clusters = f["clusters"]
        cluster_sn = f["cluster_sn"]
        cluster_contsub = f["cluster_contsub"]

        if nmax is None:
            nloc = len(clusters)
        else:
            nloc = min(nmax, len(clusters))

        delayed_tasks = [
            process_one_cluster(
                cid,
                clusters[cid],
                cluster_sn[cid],
                cluster_contsub[cid],
                f["selected_spaxels"]["x"],
                f["selected_spaxels"]["y"],
                f["selected_spaxels"]["z"],
                f["sl_low"],
                f["subw"],
            )
            for cid in range(nloc)
        ]

        futures = client.compute(delayed_tasks)

        rows = []
        with tqdm(total=len(futures), desc=f"{name} clusters") as pbar:
            for fut in as_completed(futures):
                rows.append(fut.result())
                pbar.update(1)

        rows.sort(key=lambda r: r[0])  # deterministic ordering

        fields[name]["telcatalog"] = Table(
            names=names,
            dtype=dtype,
            rows=rows
        )




###################### Dask FoF ###################################


import math
import numpy as np
import dask.array as da
import dask
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
from itertools import product
from scipy.spatial import cKDTree


# =========================
# Union-Find (Disjoint Set)
# =========================
class UnionFind:
    __slots__ = ("parent", "rank")

    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, x: int) -> int:
        p = self.parent.get(x, x)
        if p != x:
            p = self.find(p)
            self.parent[x] = p
        else:
            self.parent.setdefault(x, x)
        return p

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        rka = self.rank.get(ra, 0)
        rkb = self.rank.get(rb, 0)
        if rka < rkb:
            self.parent[ra] = rb
        elif rka > rkb:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] = rka + 1


# =========================
# Chunk geometry helpers
# =========================
def _prefix_offsets(chunks_1d: Tuple[int, ...]) -> np.ndarray:
    off = np.zeros(len(chunks_1d) + 1, dtype=np.int64)
    off[1:] = np.cumsum(np.array(chunks_1d, dtype=np.int64))
    return off

def _chunk_origin_from_offsets(offx, offy, offz, block_id: Tuple[int, int, int]) -> Tuple[int, int, int]:
    i, j, k = block_id
    return int(offx[i]), int(offy[j]), int(offz[k])

def _linear_chunk_id(block_id: Tuple[int, int, int], numblocks: Tuple[int, int, int]) -> int:
    return int(np.ravel_multi_index(block_id, numblocks))

def _make_global_cluster_id(chunk_id: int, local_label: int) -> int:
    return (np.int64(chunk_id) << np.int64(32)) | (np.int64(local_label) & np.int64(0xFFFFFFFF))


# =========================
# Group reductions for extents
# =========================
def _group_minmax(values: np.ndarray, group_idx: np.ndarray, n_groups: int) -> Tuple[np.ndarray, np.ndarray]:
    if values.size == 0 or n_groups == 0:
        return (np.empty((0,), dtype=values.dtype), np.empty((0,), dtype=values.dtype))

    order = np.argsort(group_idx, kind="mergesort")
    gi = group_idx[order]
    vv = values[order]

    starts = np.flatnonzero(np.r_[True, gi[1:] != gi[:-1]])
    gids = gi[starts]
    ends = np.r_[starts[1:], vv.size]

    if np.issubdtype(vv.dtype, np.integer):
        vmin = np.full((n_groups,), np.iinfo(vv.dtype).max, dtype=vv.dtype)
        vmax = np.full((n_groups,), np.iinfo(vv.dtype).min, dtype=vv.dtype)
    else:
        vmin = np.full((n_groups,), np.inf, dtype=vv.dtype)
        vmax = np.full((n_groups,), -np.inf, dtype=vv.dtype)

    for g, s, e in zip(gids, starts, ends):
        seg = vv[s:e]
        vmin[g] = seg.min()
        vmax[g] = seg.max()

    return vmin, vmax


# =========================
# Mapping helper (same semantics as relabel)
# =========================
def _map_ids_with_mapping(ids: np.ndarray, keys: np.ndarray, values: np.ndarray) -> np.ndarray:
    if ids.size == 0:
        return ids.astype(np.int64, copy=False)

    ids = ids.astype(np.int64, copy=False)

    if keys.size == 0:
        return ids

    idx = np.searchsorted(keys, ids)
    valid = idx < keys.size
    hit = np.zeros_like(valid, dtype=bool)
    hit[valid] = (keys[idx[valid]] == ids[valid])

    out = ids.copy()
    out[hit] = values[idx[hit]]
    return out


# =========================
# Merge per-chunk catalogs to final catalog
# =========================
def _merge_catalog_dicts(cat_list: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    ids = np.concatenate([c["id"] for c in cat_list if c["id"].size], axis=0)
    if ids.size == 0:
        return {
            "id": np.empty((0,), dtype=np.int64),
            "n_spax": np.empty((0,), dtype=np.int64),
            "xmin": np.empty((0,), dtype=np.int64),
            "xmax": np.empty((0,), dtype=np.int64),
            "ymin": np.empty((0,), dtype=np.int64),
            "ymax": np.empty((0,), dtype=np.int64),
            "zmin": np.empty((0,), dtype=np.int64),
            "zmax": np.empty((0,), dtype=np.int64),
            "flux_sum": np.empty((0,), dtype=np.float64),
            "x_flux": np.empty((0,), dtype=np.float64),
            "y_flux": np.empty((0,), dtype=np.float64),
            "z_flux": np.empty((0,), dtype=np.float64),
            "m2x": np.empty((0,), dtype=np.float64),
            "m2y": np.empty((0,), dtype=np.float64),
            "m2z": np.empty((0,), dtype=np.float64),
            "max_flux": np.empty((0,), dtype=np.float64),
        }

    def cat_concat(name, dtype=None):
        arr = np.concatenate([c[name] for c in cat_list if c["id"].size], axis=0)
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        return arr

    n_spax   = cat_concat("n_spax", np.int64)
    xmin     = cat_concat("xmin", np.int64)
    xmax     = cat_concat("xmax", np.int64)
    ymin     = cat_concat("ymin", np.int64)
    ymax     = cat_concat("ymax", np.int64)
    zmin     = cat_concat("zmin", np.int64)
    zmax     = cat_concat("zmax", np.int64)

    flux_sum = cat_concat("flux_sum", np.float64)
    sum_fx   = cat_concat("sum_fx", np.float64)
    sum_fy   = cat_concat("sum_fy", np.float64)
    sum_fz   = cat_concat("sum_fz", np.float64)
    sum_fx2  = cat_concat("sum_fx2", np.float64)
    sum_fy2  = cat_concat("sum_fy2", np.float64)
    sum_fz2  = cat_concat("sum_fz2", np.float64)
    max_flux = cat_concat("max_flux", np.float64)

    order = np.argsort(ids, kind="mergesort")
    ids_s = ids[order]
    starts = np.flatnonzero(np.r_[True, ids_s[1:] != ids_s[:-1]])

    def reduce_sum(a): return np.add.reduceat(a[order], starts)
    def reduce_min(a): return np.minimum.reduceat(a[order], starts)
    def reduce_max(a): return np.maximum.reduceat(a[order], starts)

    unique_ids = ids_s[starts]

    n_spax_g   = reduce_sum(n_spax)
    flux_sum_g = reduce_sum(flux_sum)
    sum_fx_g   = reduce_sum(sum_fx)
    sum_fy_g   = reduce_sum(sum_fy)
    sum_fz_g   = reduce_sum(sum_fz)
    sum_fx2_g  = reduce_sum(sum_fx2)
    sum_fy2_g  = reduce_sum(sum_fy2)
    sum_fz2_g  = reduce_sum(sum_fz2)

    xmin_g = reduce_min(xmin); xmax_g = reduce_max(xmax)
    ymin_g = reduce_min(ymin); ymax_g = reduce_max(ymax)
    zmin_g = reduce_min(zmin); zmax_g = reduce_max(zmax)

    max_flux_g = np.maximum.reduceat(max_flux[order], starts)

    good = flux_sum_g != 0.0
    x_flux = np.full_like(flux_sum_g, np.nan, dtype=np.float64)
    y_flux = np.full_like(flux_sum_g, np.nan, dtype=np.float64)
    z_flux = np.full_like(flux_sum_g, np.nan, dtype=np.float64)
    m2x = np.full_like(flux_sum_g, np.nan, dtype=np.float64)
    m2y = np.full_like(flux_sum_g, np.nan, dtype=np.float64)
    m2z = np.full_like(flux_sum_g, np.nan, dtype=np.float64)

    x_flux[good] = sum_fx_g[good] / flux_sum_g[good]
    y_flux[good] = sum_fy_g[good] / flux_sum_g[good]
    z_flux[good] = sum_fz_g[good] / flux_sum_g[good]

    ex2 = np.zeros_like(sum_fx2_g)
    ey2 = np.zeros_like(sum_fy2_g)
    ez2 = np.zeros_like(sum_fz2_g)
    ex2[good] = sum_fx2_g[good] / flux_sum_g[good]
    ey2[good] = sum_fy2_g[good] / flux_sum_g[good]
    ez2[good] = sum_fz2_g[good] / flux_sum_g[good]
    m2x[good] = ex2[good] - x_flux[good] ** 2
    m2y[good] = ey2[good] - y_flux[good] ** 2
    m2z[good] = ez2[good] - z_flux[good] ** 2

    return {
        "id": unique_ids.astype(np.int64, copy=False),
        "n_spax": n_spax_g.astype(np.int64, copy=False),
        "xmin": xmin_g.astype(np.int64, copy=False),
        "xmax": xmax_g.astype(np.int64, copy=False),
        "ymin": ymin_g.astype(np.int64, copy=False),
        "ymax": ymax_g.astype(np.int64, copy=False),
        "zmin": zmin_g.astype(np.int64, copy=False),
        "zmax": zmax_g.astype(np.int64, copy=False),
        "flux_sum": flux_sum_g.astype(np.float64, copy=False),
        "x_flux": x_flux,
        "y_flux": y_flux,
        "z_flux": z_flux,
        "m2x": m2x,
        "m2y": m2y,
        "m2z": m2z,
        "max_flux": max_flux_g.astype(np.float64, copy=False),
    }


# =========================
# Data container per chunk
# =========================
@dataclass
class ChunkResult:
    labels: np.ndarray
    faces: Dict[str, Tuple[np.ndarray, np.ndarray]]
    n_local: int
    chunk_id: int
    catalog_local: Dict[str, np.ndarray]


# =========================
# Local FoF inside one chunk
# =========================
def _fof_labels_for_positions(coords_local: np.ndarray, linking_length: float, kdtree_workers: int = -1) -> np.ndarray:
    n = coords_local.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.int32)

    tree = cKDTree(coords_local)
    visited = np.zeros(n, dtype=bool)
    labels = np.zeros(n, dtype=np.int32)
    cur_label = 0

    for p in range(n):
        if visited[p]:
            continue
        cur_label += 1
        stack = [p]
        visited[p] = True
        labels[p] = cur_label

        while stack:
            cur = stack.pop()
            neigh = tree.query_ball_point(coords_local[cur], r=linking_length, workers=kdtree_workers)
            for q in neigh:
                if not visited[q]:
                    visited[q] = True
                    labels[q] = cur_label
                    stack.append(q)

    return labels


def _extract_face_shell(coords_local: np.ndarray,
                        point_labels: np.ndarray,
                        chunk_shape: Tuple[int, int, int],
                        origin: Tuple[int, int, int],
                        shell: int) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    faces = {f: (np.empty((0, 3), dtype=np.int32), np.empty((0,), dtype=np.int32))
             for f in ("x0", "x1", "y0", "y1", "z0", "z1")}

    if coords_local.shape[0] == 0:
        return faces

    sh = int(shell)
    sx, sy, sz = chunk_shape
    x, y, z = coords_local[:, 0], coords_local[:, 1], coords_local[:, 2]

    masks = {
        "x0": x < sh,
        "x1": x >= (sx - sh),
        "y0": y < sh,
        "y1": y >= (sy - sh),
        "z0": z < sh,
        "z1": z >= (sz - sh),
    }

    ox, oy, oz = origin
    for key, m in masks.items():
        c = coords_local[m].astype(np.int32, copy=False)
        if c.shape[0] == 0:
            continue
        g = c.copy()
        g[:, 0] += ox
        g[:, 1] += oy
        g[:, 2] += oz
        faces[key] = (g, point_labels[m].astype(np.int32, copy=False))

    return faces


def _catalog_for_chunk_points(coords_local: np.ndarray,
                              point_labels: np.ndarray,
                              origin: Tuple[int, int, int],
                              flux_block: np.ndarray,
                              sn_block: np.ndarray) -> Dict[str, np.ndarray]:
    """
    coords order = (z,y,x)
    """

    npts = coords_local.shape[0]

    if npts == 0:
        return {
            "local_label": np.empty((0,), dtype=np.int64),
            "n_spax": np.empty((0,), dtype=np.int64),

            "xmin": np.empty((0,), dtype=np.int64),
            "xmax": np.empty((0,), dtype=np.int64),
            "ymin": np.empty((0,), dtype=np.int64),
            "ymax": np.empty((0,), dtype=np.int64),
            "zmin": np.empty((0,), dtype=np.int64),
            "zmax": np.empty((0,), dtype=np.int64),

            "flux_sum": np.empty((0,), dtype=np.float64),

            "sum_fx": np.empty((0,), dtype=np.float64),
            "sum_fy": np.empty((0,), dtype=np.float64),
            "sum_fz": np.empty((0,), dtype=np.float64),

            "sum_fx2": np.empty((0,), dtype=np.float64),
            "sum_fy2": np.empty((0,), dtype=np.float64),
            "sum_fz2": np.empty((0,), dtype=np.float64),

            "max_flux": np.empty((0,), dtype=np.float64),
            "max_sn": np.empty((0,), dtype=np.float64),
        }

    n_local = int(point_labels.max())
    group_idx = point_labels.astype(np.int64) - 1

    oz, oy, ox = origin

    gz = coords_local[:,0].astype(np.int64) + oz
    gy = coords_local[:,1].astype(np.int64) + oy
    gx = coords_local[:,2].astype(np.int64) + ox

    fv = flux_block[coords_local[:,0],
                    coords_local[:,1],
                    coords_local[:,2]].astype(np.float64, copy=False)

    snv = sn_block[coords_local[:,0],
                   coords_local[:,1],
                   coords_local[:,2]].astype(np.float64, copy=False)

    n_spax = np.bincount(group_idx, minlength=n_local).astype(np.int64)

    flux_sum = np.bincount(group_idx, weights=fv, minlength=n_local)

    sum_fx  = np.bincount(group_idx, weights=fv*gx, minlength=n_local)
    sum_fy  = np.bincount(group_idx, weights=fv*gy, minlength=n_local)
    sum_fz  = np.bincount(group_idx, weights=fv*gz, minlength=n_local)

    sum_fx2 = np.bincount(group_idx, weights=fv*gx*gx, minlength=n_local)
    sum_fy2 = np.bincount(group_idx, weights=fv*gy*gy, minlength=n_local)
    sum_fz2 = np.bincount(group_idx, weights=fv*gz*gz, minlength=n_local)

    xmin,xmax = _group_minmax(gx, group_idx, n_local)
    ymin,ymax = _group_minmax(gy, group_idx, n_local)
    zmin,zmax = _group_minmax(gz, group_idx, n_local)

    # max flux + max sn
    order = np.argsort(group_idx, kind="mergesort")
    gi = group_idx[order]

    starts = np.flatnonzero(np.r_[True, gi[1:]!=gi[:-1]])
    gids_present = gi[starts]

    max_flux = np.full((n_local,), -np.inf)
    max_sn   = np.full((n_local,), -np.inf)

    max_flux[gids_present] = np.maximum.reduceat(fv[order], starts)
    max_sn[gids_present]   = np.maximum.reduceat(snv[order], starts)

    return {
        "local_label": np.arange(n_local,dtype=np.int64)+1,
        "n_spax": n_spax,

        "xmin": xmin,
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax,
        "zmin": zmin,
        "zmax": zmax,

        "flux_sum": flux_sum,

        "sum_fx": sum_fx,
        "sum_fy": sum_fy,
        "sum_fz": sum_fz,

        "sum_fx2": sum_fx2,
        "sum_fy2": sum_fy2,
        "sum_fz2": sum_fz2,

        "max_flux": max_flux,
        "max_sn": max_sn,
    }




def _process_one_chunk(block_bool: np.ndarray,
                       block_flux: np.ndarray,
                       block_sn: np.ndarray,   # currently unused, wired in for future
                       *,
                       origin: Tuple[int, int, int],
                       chunk_id: int,
                       linking_length: float,
                       shell: int,
                       kdtree_workers: int = -1) -> ChunkResult:
    coords = np.argwhere(block_bool)
    point_labels = _fof_labels_for_positions(coords, linking_length, kdtree_workers=kdtree_workers)

    labels = np.zeros(block_bool.shape, dtype=np.int32)
    if coords.shape[0] > 0:
        labels[coords[:, 0], coords[:, 1], coords[:, 2]] = point_labels

    n_local = int(point_labels.max()) if point_labels.size else 0

    faces_local = _extract_face_shell(coords, point_labels, block_bool.shape, origin, shell)

    faces_global: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for face, (gcoords, labs_local) in faces_local.items():
        if gcoords.shape[0] == 0:
            faces_global[face] = (gcoords, np.empty((0,), dtype=np.int64))
            continue
        # vectorized packing for face gids
        gids = (np.int64(chunk_id) << np.int64(32)) | (labs_local.astype(np.int64) & np.int64(0xFFFFFFFF))
        faces_global[face] = (gcoords, gids.astype(np.int64, copy=False))

    # Local catalog aggregates
    cat = _catalog_for_chunk_points(coords, point_labels, origin, block_flux, block_sn)


    # vectorized packing for catalog gids (performance improvement)
    ll = cat["local_label"].astype(np.int64, copy=False)
    if ll.size:
        gids_local = (np.int64(chunk_id) << np.int64(32)) | (ll & np.int64(0xFFFFFFFF))
    else:
        gids_local = np.empty((0,), dtype=np.int64)

    cat["gid"] = gids_local
    cat.pop("local_label", None)

    return ChunkResult(labels=labels, faces=faces_global, n_local=n_local, chunk_id=chunk_id, catalog_local=cat)


# =========================
# Cross-chunk edge building
# =========================
def _edges_between_faces(faceA: Tuple[np.ndarray, np.ndarray],
                         faceB: Tuple[np.ndarray, np.ndarray],
                         linking_length: float,
                         kdtree_workers: int = -1) -> np.ndarray:
    coordsA, gidsA = faceA
    coordsB, gidsB = faceB
    if coordsA.shape[0] == 0 or coordsB.shape[0] == 0:
        return np.empty((0, 2), dtype=np.int64)

    treeB = cKDTree(coordsB.astype(np.float32, copy=False))
    edges = []

    for i in range(coordsA.shape[0]):
        neigh = treeB.query_ball_point(coordsA[i].astype(np.float32, copy=False),
                                       r=linking_length,
                                       workers=kdtree_workers)
        if not neigh:
            continue
        ga = int(gidsA[i])
        for j in neigh:
            gb = int(gidsB[j])
            if ga != gb:
                if ga < gb:
                    edges.append((ga, gb))
                else:
                    edges.append((gb, ga))

    if not edges:
        return np.empty((0, 2), dtype=np.int64)

    e = np.array(edges, dtype=np.int64)
    e = np.unique(e, axis=0)
    return e


def _concat_edges(*edge_arrays: np.ndarray) -> np.ndarray:
    nonempty = [e for e in edge_arrays if e is not None and e.size]
    if not nonempty:
        return np.empty((0, 2), dtype=np.int64)
    out = np.vstack(nonempty)
    out = np.unique(out, axis=0)
    return out


def _build_mapping_from_edges(edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if edges.size == 0:
        return (np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64))

    uf = UnionFind()
    for a, b in edges:
        uf.union(int(a), int(b))

    nodes = np.unique(edges.reshape(-1)).astype(np.int64)
    root_to_label = {}
    next_label = 1
    vals = np.empty(nodes.shape[0], dtype=np.int64)

    for i, n in enumerate(nodes):
        r = uf.find(int(n))
        lab = root_to_label.get(r)
        if lab is None:
            lab = next_label
            root_to_label[r] = lab
            next_label += 1
        vals[i] = lab

    order = np.argsort(nodes)
    keys = nodes[order]
    values = vals[order]
    return keys, values


# =========================
# Relabel chunk to global ids
# =========================
def _relabel_chunk(chunk_res, keys, values):
    lab_local = chunk_res.labels
    out = np.zeros(lab_local.shape, dtype=np.int64)

    m = lab_local > 0
    if not np.any(m):
        return out

    packed = (
        (np.int64(chunk_res.chunk_id) << np.int64(32))
        | (lab_local[m].astype(np.int64) & np.int64(0xFFFFFFFF))
    )

    if keys.size == 0:
        out[m] = packed
        return out

    idx = np.searchsorted(keys, packed)
    valid = idx < keys.size
    hit = np.zeros_like(valid, dtype=bool)
    hit[valid] = (keys[idx[valid]] == packed[valid])

    out_m = out[m]
    out_m[hit] = values[idx[hit]]
    out_m[~hit] = packed[~hit]
    out[m] = out_m
    return out


def _finalize_chunk_catalog(chunk_res: ChunkResult, keys: np.ndarray, values: np.ndarray) -> Dict[str, np.ndarray]:
    cat = chunk_res.catalog_local
    out = {k: v for k, v in cat.items() if k != "gid"}
    out["id"] = _map_ids_with_mapping(cat["gid"], keys, values)
    return out


# =========================
# Public API
# =========================
def fof_boolean_zarr_hpc_with_catalog(arr_bool,
                                      flux,
                                      sn,
                                      linking_length: float,
                                      *,
                                      shell: Optional[int] = None,
                                      kdtree_workers: int = -1,
                                      output_chunks: Optional[Tuple[int, int, int]] = None):
    darr = da.asarray(arr_bool)
    if darr.ndim != 3:
        raise ValueError("Expected a 3D array")

    dflux = da.asarray(flux)
    dsn = da.asarray(sn)

    # align chunking lazily
    if dflux.chunks != darr.chunks:
        dflux = dflux.rechunk(darr.chunks)
    if dsn.chunks != darr.chunks:
        dsn = dsn.rechunk(darr.chunks)

    if shell is None:
        shell = int(math.ceil(float(linking_length)))
    if shell < 1:
        raise ValueError("shell must be >= 1")

    chunks = darr.chunks
    numblocks = darr.numblocks

    offx = _prefix_offsets(chunks[0])
    offy = _prefix_offsets(chunks[1])
    offz = _prefix_offsets(chunks[2])

    delayed_blocks_bool = darr.to_delayed().ravel()
    delayed_blocks_flux = dflux.to_delayed().ravel()
    delayed_blocks_sn   = dsn.to_delayed().ravel()

    block_ids = list(product(range(numblocks[0]), range(numblocks[1]), range(numblocks[2])))

    # Stage A
    chunk_results = []
    for idx, bid in enumerate(block_ids):
        origin = _chunk_origin_from_offsets(offx, offy, offz, bid)
        cid = _linear_chunk_id(bid, numblocks)
        chunk_results.append(
            dask.delayed(_process_one_chunk)(
                delayed_blocks_bool[idx],
                delayed_blocks_flux[idx],
                delayed_blocks_sn[idx],
                origin=origin,
                chunk_id=cid,
                linking_length=linking_length,
                shell=shell,
                kdtree_workers=kdtree_workers
            )
        )

    # Stage B: cross-chunk edges
    def in_bounds(b):
        return (0 <= b[0] < numblocks[0]) and (0 <= b[1] < numblocks[1]) and (0 <= b[2] < numblocks[2])

    idx_of = {bid: i for i, bid in enumerate(block_ids)}
    neighbor_dirs = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    dir_faces = {
        (1, 0, 0): ("x1", "x0"),
        (0, 1, 0): ("y1", "y0"),
        (0, 0, 1): ("z1", "z0"),
    }

    edge_tasks = []
    for bid in block_ids:
        for d in neighbor_dirs:
            nb = (bid[0] + d[0], bid[1] + d[1], bid[2] + d[2])
            if not in_bounds(nb):
                continue
            ia = idx_of[bid]
            ib = idx_of[nb]
            fa, fb = dir_faces[d]

            edge_tasks.append(
                dask.delayed(lambda ra, rb, fa=fa, fb=fb:
                             _edges_between_faces(ra.faces[fa], rb.faces[fb],
                                                 linking_length=linking_length,
                                                 kdtree_workers=kdtree_workers)
                             )(chunk_results[ia], chunk_results[ib])
            )

    all_edges = dask.delayed(_concat_edges)(*edge_tasks)
    mapping_kv = dask.delayed(_build_mapping_from_edges)(all_edges)  # (keys, values)

    # Stage C: relabel chunks into a global label array
    relabeled_blocks = []
    for idx, bid in enumerate(block_ids):
        sx = chunks[0][bid[0]]
        sy = chunks[1][bid[1]]
        sz = chunks[2][bid[2]]
        shape = (sx, sy, sz)

        relabeled_blocks.append(
            da.from_delayed(
                dask.delayed(_relabel_chunk)(chunk_results[idx], mapping_kv[0], mapping_kv[1]),
                shape=shape,
                dtype=np.int64
            )
        )

    grid_blocks = [[[None for _ in range(numblocks[2])]
                    for _ in range(numblocks[1])]
                   for _ in range(numblocks[0])]

    for bid in block_ids:
        grid_blocks[bid[0]][bid[1]][bid[2]] = relabeled_blocks[idx_of[bid]]

    labels = da.block(grid_blocks)

    if output_chunks is not None:
        labels = labels.rechunk(output_chunks)

    # Catalog: finalize ids then merge across chunks
    chunk_cats_final = [
        dask.delayed(_finalize_chunk_catalog)(cr, mapping_kv[0], mapping_kv[1])
        for cr in chunk_results
    ]
    catalog = dask.delayed(_merge_catalog_dicts)(chunk_cats_final)

    return labels, catalog


# -------------------------
# Example usage:
# labels, cat_delayed = fof_boolean_zarr_hpc_with_catalog(arr, flux_cube, sn_cube, linking_length=1.5)
# cat = dask.compute(cat_delayed)[0]
# from astropy.table import Table
# tab = Table(cat)
# -------------------------

from astropy.table import Table
import numpy as np


def catalog_dict_to_astropy_table(cat: dict) -> Table:
    """
    Convert the FoF catalog dictionary into an Astropy Table.

    Parameters
    ----------
    cat : dict
        Output from dask.compute(cat_delayed)[0]

    Returns
    -------
    tab : astropy.table.Table
    """

    # --- ensure deterministic ordering (optional but recommended)
    if "id" in cat and cat["id"].size:
        order = np.argsort(cat["id"])
    else:
        order = slice(None)

    tab = Table()

    # copy all fields into table columns
    for key, val in cat.items():
        tab[key] = val[order]

    # optional: attach units / metadata (edit as needed)
    tab["x_flux"].unit = "pixel"
    tab["y_flux"].unit = "pixel"
    tab["z_flux"].unit = "pixel"

    return tab

from astropy.table import Table
import numpy as np

from astropy.table import Table
from astropy.wcs import WCS
import numpy as np


def catalog_dict_to_astropy_table(cat: dict, header=None) -> Table:
    """
    Convert FoF catalog dictionary into an Astropy Table
    and optionally convert pixel coordinates to world coordinates.

    Parameters
    ----------
    cat : dict
        Output from dask.compute(cat_delayed)[0]

    header : astropy.io.fits.Header or astropy.wcs.WCS, optional
        If provided, pixel coordinates will be converted to world
        coordinates using WCS.

    Returns
    -------
    tab : astropy.table.Table
    """

    # -----------------------------
    # deterministic ordering
    # -----------------------------
    if "id" in cat and cat["id"].size:
        order = np.argsort(cat["id"])
    else:
        order = slice(None)

    tab = Table()

    for key, val in cat.items():
        tab[key] = val[order]

    # pixel units
    if "x_flux" in tab.colnames:
        tab["x_flux"].unit = "pixel"
        tab["y_flux"].unit = "pixel"
        tab["z_flux"].unit = "pixel"

    # -----------------------------
    # WCS conversion
    # -----------------------------
    if header is not None:

        # accept header or WCS object
        wcs = header if isinstance(header, WCS) else WCS(header)

        # IMPORTANT:
        # numpy index order = (z,y,x)
        # astropy WCS expects (x,y,z)

        if {"x_flux","y_flux","z_flux"}.issubset(tab.colnames):

            x = np.asarray(tab["x_flux"], dtype=float)
            y = np.asarray(tab["y_flux"], dtype=float)
            z = np.asarray(tab["z_flux"], dtype=float)

            # origin=0 for numpy indexing
            world = wcs.all_pix2world(x, y, z, 0)

            # world is tuple/list of arrays
            tab["world_x"] = world[0]
            tab["world_y"] = world[1]
            tab["world_z"] = world[2]

        # Optional: also convert extents if present
        extent_fields = [
            ("xmin","ymin","zmin","world_xmin","world_ymin","world_zmin"),
            ("xmax","ymax","zmax","world_xmax","world_ymax","world_zmax"),
        ]

        for xf,yf,zf,wx,wy,wz in extent_fields:

            if {xf,yf,zf}.issubset(tab.colnames):

                world = wcs.all_pix2world(
                    np.asarray(tab[xf],float),
                    np.asarray(tab[yf],float),
                    np.asarray(tab[zf],float),
                    0
                )

                tab[wx] = world[0]
                tab[wy] = world[1]
                tab[wz] = world[2]

    return tab

###############################
import math
import numpy as np
import dask.array as da
import dask
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
from itertools import product
from scipy.spatial import cKDTree


# =========================
# Union-Find (Disjoint Set)
# =========================
class UnionFind:
    __slots__ = ("parent", "rank")

    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, x: int) -> int:
        p = self.parent.get(x, x)
        if p != x:
            p = self.find(p)
            self.parent[x] = p
        else:
            self.parent.setdefault(x, x)
        return p

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        rka = self.rank.get(ra, 0)
        rkb = self.rank.get(rb, 0)
        if rka < rkb:
            self.parent[ra] = rb
        elif rka > rkb:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] = rka + 1


# =========================
# Chunk geometry helpers
# =========================
def _prefix_offsets(chunks_1d: Tuple[int, ...]) -> np.ndarray:
    off = np.zeros(len(chunks_1d) + 1, dtype=np.int64)
    off[1:] = np.cumsum(np.array(chunks_1d, dtype=np.int64))
    return off

def _chunk_origin_from_offsets(offx, offy, offz, block_id: Tuple[int, int, int]) -> Tuple[int, int, int]:
    i, j, k = block_id
    return int(offx[i]), int(offy[j]), int(offz[k])

def _linear_chunk_id(block_id: Tuple[int, int, int], numblocks: Tuple[int, int, int]) -> int:
    return int(np.ravel_multi_index(block_id, numblocks))

def _make_global_cluster_id(chunk_id: int, local_label: int) -> int:
    return (np.int64(chunk_id) << np.int64(32)) | (np.int64(local_label) & np.int64(0xFFFFFFFF))


# =========================
# Group reductions for extents
# =========================
def _group_minmax(values: np.ndarray, group_idx: np.ndarray, n_groups: int) -> Tuple[np.ndarray, np.ndarray]:
    if values.size == 0 or n_groups == 0:
        return (np.empty((0,), dtype=values.dtype), np.empty((0,), dtype=values.dtype))

    order = np.argsort(group_idx, kind="mergesort")
    gi = group_idx[order]
    vv = values[order]

    starts = np.flatnonzero(np.r_[True, gi[1:] != gi[:-1]])
    gids = gi[starts]
    ends = np.r_[starts[1:], vv.size]

    if np.issubdtype(vv.dtype, np.integer):
        vmin = np.full((n_groups,), np.iinfo(vv.dtype).max, dtype=vv.dtype)
        vmax = np.full((n_groups,), np.iinfo(vv.dtype).min, dtype=vv.dtype)
    else:
        vmin = np.full((n_groups,), np.inf, dtype=vv.dtype)
        vmax = np.full((n_groups,), -np.inf, dtype=vv.dtype)

    for g, s, e in zip(gids, starts, ends):
        seg = vv[s:e]
        vmin[g] = seg.min()
        vmax[g] = seg.max()

    return vmin, vmax


# =========================
# Mapping helper (same semantics as relabel)
# =========================
def _map_ids_with_mapping(ids: np.ndarray, keys: np.ndarray, values: np.ndarray) -> np.ndarray:
    if ids.size == 0:
        return ids.astype(np.int64, copy=False)

    ids = ids.astype(np.int64, copy=False)

    if keys.size == 0:
        return ids

    idx = np.searchsorted(keys, ids)
    valid = idx < keys.size
    hit = np.zeros_like(valid, dtype=bool)
    hit[valid] = (keys[idx[valid]] == ids[valid])

    out = ids.copy()
    out[hit] = values[idx[hit]]
    return out


# =========================
# Merge per-chunk catalogs to final catalog
# =========================
def _merge_catalog_dicts(cat_list: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    ids = np.concatenate([c["id"] for c in cat_list if c["id"].size], axis=0)
    if ids.size == 0:
        return {
            "id": np.empty((0,), dtype=np.int64),
            "n_spax": np.empty((0,), dtype=np.int64),
            "xmin": np.empty((0,), dtype=np.int64),
            "xmax": np.empty((0,), dtype=np.int64),
            "ymin": np.empty((0,), dtype=np.int64),
            "ymax": np.empty((0,), dtype=np.int64),
            "zmin": np.empty((0,), dtype=np.int64),
            "zmax": np.empty((0,), dtype=np.int64),
            "flux_sum": np.empty((0,), dtype=np.float64),
            "x_flux": np.empty((0,), dtype=np.float64),
            "y_flux": np.empty((0,), dtype=np.float64),
            "z_flux": np.empty((0,), dtype=np.float64),
            "m2x": np.empty((0,), dtype=np.float64),
            "m2y": np.empty((0,), dtype=np.float64),
            "m2z": np.empty((0,), dtype=np.float64),
            "max_flux": np.empty((0,), dtype=np.float64),
            "max_sn": np.empty((0,), dtype=np.float64),  # NEW
        }

    def cat_concat(name, dtype=None):
        arr = np.concatenate([c[name] for c in cat_list if c["id"].size], axis=0)
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        return arr

    n_spax   = cat_concat("n_spax", np.int64)
    xmin     = cat_concat("xmin", np.int64)
    xmax     = cat_concat("xmax", np.int64)
    ymin     = cat_concat("ymin", np.int64)
    ymax     = cat_concat("ymax", np.int64)
    zmin     = cat_concat("zmin", np.int64)
    zmax     = cat_concat("zmax", np.int64)

    flux_sum = cat_concat("flux_sum", np.float64)
    sum_fx   = cat_concat("sum_fx", np.float64)
    sum_fy   = cat_concat("sum_fy", np.float64)
    sum_fz   = cat_concat("sum_fz", np.float64)
    sum_fx2  = cat_concat("sum_fx2", np.float64)
    sum_fy2  = cat_concat("sum_fy2", np.float64)
    sum_fz2  = cat_concat("sum_fz2", np.float64)

    max_flux = cat_concat("max_flux", np.float64)
    max_sn   = cat_concat("max_sn", np.float64)  # NEW

    order = np.argsort(ids, kind="mergesort")
    ids_s = ids[order]
    starts = np.flatnonzero(np.r_[True, ids_s[1:] != ids_s[:-1]])

    def reduce_sum(a): return np.add.reduceat(a[order], starts)
    def reduce_min(a): return np.minimum.reduceat(a[order], starts)
    def reduce_max(a): return np.maximum.reduceat(a[order], starts)

    unique_ids = ids_s[starts]

    n_spax_g   = reduce_sum(n_spax)
    flux_sum_g = reduce_sum(flux_sum)
    sum_fx_g   = reduce_sum(sum_fx)
    sum_fy_g   = reduce_sum(sum_fy)
    sum_fz_g   = reduce_sum(sum_fz)
    sum_fx2_g  = reduce_sum(sum_fx2)
    sum_fy2_g  = reduce_sum(sum_fy2)
    sum_fz2_g  = reduce_sum(sum_fz2)

    xmin_g = reduce_min(xmin); xmax_g = reduce_max(xmax)
    ymin_g = reduce_min(ymin); ymax_g = reduce_max(ymax)
    zmin_g = reduce_min(zmin); zmax_g = reduce_max(zmax)

    max_flux_g = reduce_max(max_flux)
    max_sn_g   = reduce_max(max_sn)  # NEW

    good = flux_sum_g != 0.0
    x_flux = np.full_like(flux_sum_g, np.nan, dtype=np.float64)
    y_flux = np.full_like(flux_sum_g, np.nan, dtype=np.float64)
    z_flux = np.full_like(flux_sum_g, np.nan, dtype=np.float64)
    m2x = np.full_like(flux_sum_g, np.nan, dtype=np.float64)
    m2y = np.full_like(flux_sum_g, np.nan, dtype=np.float64)
    m2z = np.full_like(flux_sum_g, np.nan, dtype=np.float64)

    x_flux[good] = sum_fx_g[good] / flux_sum_g[good]
    y_flux[good] = sum_fy_g[good] / flux_sum_g[good]
    z_flux[good] = sum_fz_g[good] / flux_sum_g[good]

    ex2 = np.zeros_like(sum_fx2_g)
    ey2 = np.zeros_like(sum_fy2_g)
    ez2 = np.zeros_like(sum_fz2_g)
    ex2[good] = sum_fx2_g[good] / flux_sum_g[good]
    ey2[good] = sum_fy2_g[good] / flux_sum_g[good]
    ez2[good] = sum_fz2_g[good] / flux_sum_g[good]
    m2x[good] = ex2[good] - x_flux[good] ** 2
    m2y[good] = ey2[good] - y_flux[good] ** 2
    m2z[good] = ez2[good] - z_flux[good] ** 2

    return {
        "id": unique_ids.astype(np.int64, copy=False),
        "n_spax": n_spax_g.astype(np.int64, copy=False),
        "xmin": xmin_g.astype(np.int64, copy=False),
        "xmax": xmax_g.astype(np.int64, copy=False),
        "ymin": ymin_g.astype(np.int64, copy=False),
        "ymax": ymax_g.astype(np.int64, copy=False),
        "zmin": zmin_g.astype(np.int64, copy=False),
        "zmax": zmax_g.astype(np.int64, copy=False),
        "flux_sum": flux_sum_g.astype(np.float64, copy=False),
        "x_flux": x_flux,
        "y_flux": y_flux,
        "z_flux": z_flux,
        "m2x": m2x,
        "m2y": m2y,
        "m2z": m2z,
        "max_flux": max_flux_g.astype(np.float64, copy=False),
        "max_sn": max_sn_g.astype(np.float64, copy=False),  # NEW
    }


# =========================
# Data container per chunk
# =========================
@dataclass
class ChunkResult:
    labels: np.ndarray
    faces: Dict[str, Tuple[np.ndarray, np.ndarray]]
    n_local: int
    chunk_id: int
    catalog_local: Dict[str, np.ndarray]


# =========================
# Local FoF inside one chunk
# =========================
def _fof_labels_for_positions(coords_local: np.ndarray, linking_length: float, kdtree_workers: int = -1) -> np.ndarray:
    n = coords_local.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.int32)

    tree = cKDTree(coords_local)
    visited = np.zeros(n, dtype=bool)
    labels = np.zeros(n, dtype=np.int32)
    cur_label = 0

    for p in range(n):
        if visited[p]:
            continue
        cur_label += 1
        stack = [p]
        visited[p] = True
        labels[p] = cur_label

        while stack:
            cur = stack.pop()
            neigh = tree.query_ball_point(coords_local[cur], r=linking_length, workers=kdtree_workers)
            for q in neigh:
                if not visited[q]:
                    visited[q] = True
                    labels[q] = cur_label
                    stack.append(q)

    return labels


def _extract_face_shell(coords_local: np.ndarray,
                        point_labels: np.ndarray,
                        chunk_shape: Tuple[int, int, int],
                        origin: Tuple[int, int, int],
                        shell: int) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    faces = {f: (np.empty((0, 3), dtype=np.int32), np.empty((0,), dtype=np.int32))
             for f in ("x0", "x1", "y0", "y1", "z0", "z1")}

    if coords_local.shape[0] == 0:
        return faces

    sh = int(shell)
    sx, sy, sz = chunk_shape
    x, y, z = coords_local[:, 0], coords_local[:, 1], coords_local[:, 2]

    masks = {
        "x0": x < sh,
        "x1": x >= (sx - sh),
        "y0": y < sh,
        "y1": y >= (sy - sh),
        "z0": z < sh,
        "z1": z >= (sz - sh),
    }

    ox, oy, oz = origin
    for key, m in masks.items():
        c = coords_local[m].astype(np.int32, copy=False)
        if c.shape[0] == 0:
            continue
        g = c.copy()
        g[:, 0] += ox
        g[:, 1] += oy
        g[:, 2] += oz
        faces[key] = (g, point_labels[m].astype(np.int32, copy=False))

    return faces


def _catalog_for_chunk_points(coords_local: np.ndarray,
                              point_labels: np.ndarray,
                              origin: Tuple[int, int, int],
                              flux_block: np.ndarray,
                              sn_block: np.ndarray) -> Dict[str, np.ndarray]:
    """
    coords order = (z,y,x)
    """
    npts = coords_local.shape[0]

    if npts == 0:
        return {
            "local_label": np.empty((0,), dtype=np.int64),
            "n_spax": np.empty((0,), dtype=np.int64),

            "xmin": np.empty((0,), dtype=np.int64),
            "xmax": np.empty((0,), dtype=np.int64),
            "ymin": np.empty((0,), dtype=np.int64),
            "ymax": np.empty((0,), dtype=np.int64),
            "zmin": np.empty((0,), dtype=np.int64),
            "zmax": np.empty((0,), dtype=np.int64),

            "flux_sum": np.empty((0,), dtype=np.float64),

            "sum_fx": np.empty((0,), dtype=np.float64),
            "sum_fy": np.empty((0,), dtype=np.float64),
            "sum_fz": np.empty((0,), dtype=np.float64),

            "sum_fx2": np.empty((0,), dtype=np.float64),
            "sum_fy2": np.empty((0,), dtype=np.float64),
            "sum_fz2": np.empty((0,), dtype=np.float64),

            "max_flux": np.empty((0,), dtype=np.float64),
            "max_sn": np.empty((0,), dtype=np.float64),
        }

    n_local = int(point_labels.max())
    group_idx = point_labels.astype(np.int64) - 1

    oz, oy, ox = origin

    gz = coords_local[:, 0].astype(np.int64) + oz
    gy = coords_local[:, 1].astype(np.int64) + oy
    gx = coords_local[:, 2].astype(np.int64) + ox

    fv = flux_block[coords_local[:, 0],
                    coords_local[:, 1],
                    coords_local[:, 2]].astype(np.float64, copy=False)

    snv = sn_block[coords_local[:, 0],
                   coords_local[:, 1],
                   coords_local[:, 2]].astype(np.float64, copy=False)

    n_spax = np.bincount(group_idx, minlength=n_local).astype(np.int64)

    flux_sum = np.bincount(group_idx, weights=fv, minlength=n_local)

    sum_fx  = np.bincount(group_idx, weights=fv * gx, minlength=n_local)
    sum_fy  = np.bincount(group_idx, weights=fv * gy, minlength=n_local)
    sum_fz  = np.bincount(group_idx, weights=fv * gz, minlength=n_local)

    sum_fx2 = np.bincount(group_idx, weights=fv * gx * gx, minlength=n_local)
    sum_fy2 = np.bincount(group_idx, weights=fv * gy * gy, minlength=n_local)
    sum_fz2 = np.bincount(group_idx, weights=fv * gz * gz, minlength=n_local)

    xmin, xmax = _group_minmax(gx, group_idx, n_local)
    ymin, ymax = _group_minmax(gy, group_idx, n_local)
    zmin, zmax = _group_minmax(gz, group_idx, n_local)

    # max flux + max sn
    order = np.argsort(group_idx, kind="mergesort")
    gi = group_idx[order]

    starts = np.flatnonzero(np.r_[True, gi[1:] != gi[:-1]])
    gids_present = gi[starts]

    max_flux = np.full((n_local,), -np.inf, dtype=np.float64)
    max_sn   = np.full((n_local,), -np.inf, dtype=np.float64)

    max_flux[gids_present] = np.maximum.reduceat(fv[order], starts)
    max_sn[gids_present]   = np.maximum.reduceat(snv[order], starts)

    return {
        "local_label": np.arange(n_local, dtype=np.int64) + 1,
        "n_spax": n_spax,

        "xmin": xmin,
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax,
        "zmin": zmin,
        "zmax": zmax,

        "flux_sum": flux_sum,

        "sum_fx": sum_fx,
        "sum_fy": sum_fy,
        "sum_fz": sum_fz,

        "sum_fx2": sum_fx2,
        "sum_fy2": sum_fy2,
        "sum_fz2": sum_fz2,

        "max_flux": max_flux,
        "max_sn": max_sn,
    }


def _process_one_chunk(block_bool: np.ndarray,
                       block_flux: np.ndarray,
                       block_sn: np.ndarray,
                       *,
                       origin: Tuple[int, int, int],
                       chunk_id: int,
                       linking_length: float,
                       shell: int,
                       kdtree_workers: int = -1) -> ChunkResult:
    coords = np.argwhere(block_bool)
    point_labels = _fof_labels_for_positions(coords, linking_length, kdtree_workers=kdtree_workers)

    labels = np.zeros(block_bool.shape, dtype=np.int32)
    if coords.shape[0] > 0:
        labels[coords[:, 0], coords[:, 1], coords[:, 2]] = point_labels

    n_local = int(point_labels.max()) if point_labels.size else 0

    faces_local = _extract_face_shell(coords, point_labels, block_bool.shape, origin, shell)

    faces_global: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for face, (gcoords, labs_local) in faces_local.items():
        if gcoords.shape[0] == 0:
            faces_global[face] = (gcoords, np.empty((0,), dtype=np.int64))
            continue
        gids = (np.int64(chunk_id) << np.int64(32)) | (labs_local.astype(np.int64) & np.int64(0xFFFFFFFF))
        faces_global[face] = (gcoords, gids.astype(np.int64, copy=False))

    cat = _catalog_for_chunk_points(coords, point_labels, origin, block_flux, block_sn)

    ll = cat["local_label"].astype(np.int64, copy=False)
    if ll.size:
        gids_local = (np.int64(chunk_id) << np.int64(32)) | (ll & np.int64(0xFFFFFFFF))
    else:
        gids_local = np.empty((0,), dtype=np.int64)

    cat["gid"] = gids_local
    cat.pop("local_label", None)

    return ChunkResult(labels=labels, faces=faces_global, n_local=n_local, chunk_id=chunk_id, catalog_local=cat)


# =========================
# Cross-chunk edge building
# =========================
def _edges_between_faces(faceA: Tuple[np.ndarray, np.ndarray],
                         faceB: Tuple[np.ndarray, np.ndarray],
                         linking_length: float,
                         kdtree_workers: int = -1) -> np.ndarray:
    coordsA, gidsA = faceA
    coordsB, gidsB = faceB
    if coordsA.shape[0] == 0 or coordsB.shape[0] == 0:
        return np.empty((0, 2), dtype=np.int64)

    treeB = cKDTree(coordsB.astype(np.float32, copy=False))
    edges = []

    for i in range(coordsA.shape[0]):
        neigh = treeB.query_ball_point(coordsA[i].astype(np.float32, copy=False),
                                       r=linking_length,
                                       workers=kdtree_workers)
        if not neigh:
            continue
        ga = int(gidsA[i])
        for j in neigh:
            gb = int(gidsB[j])
            if ga != gb:
                if ga < gb:
                    edges.append((ga, gb))
                else:
                    edges.append((gb, ga))

    if not edges:
        return np.empty((0, 2), dtype=np.int64)

    e = np.array(edges, dtype=np.int64)
    e = np.unique(e, axis=0)
    return e


def _concat_edges(*edge_arrays: np.ndarray) -> np.ndarray:
    nonempty = [e for e in edge_arrays if e is not None and e.size]
    if not nonempty:
        return np.empty((0, 2), dtype=np.int64)
    out = np.vstack(nonempty)
    out = np.unique(out, axis=0)
    return out


def _build_mapping_from_edges(edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if edges.size == 0:
        return (np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64))

    uf = UnionFind()
    for a, b in edges:
        uf.union(int(a), int(b))

    nodes = np.unique(edges.reshape(-1)).astype(np.int64)
    root_to_label = {}
    next_label = 1
    vals = np.empty(nodes.shape[0], dtype=np.int64)

    for i, n in enumerate(nodes):
        r = uf.find(int(n))
        lab = root_to_label.get(r)
        if lab is None:
            lab = next_label
            root_to_label[r] = lab
            next_label += 1
        vals[i] = lab

    order = np.argsort(nodes)
    keys = nodes[order]
    values = vals[order]
    return keys, values


# =========================
# Relabel chunk to global ids
# =========================
def _relabel_chunk(chunk_res, keys, values):
    lab_local = chunk_res.labels
    out = np.zeros(lab_local.shape, dtype=np.int64)

    m = lab_local > 0
    if not np.any(m):
        return out

    packed = (
        (np.int64(chunk_res.chunk_id) << np.int64(32))
        | (lab_local[m].astype(np.int64) & np.int64(0xFFFFFFFF))
    )

    if keys.size == 0:
        out[m] = packed
        return out

    idx = np.searchsorted(keys, packed)
    valid = idx < keys.size
    hit = np.zeros_like(valid, dtype=bool)
    hit[valid] = (keys[idx[valid]] == packed[valid])

    out_m = out[m]
    out_m[hit] = values[idx[hit]]
    out_m[~hit] = packed[~hit]
    out[m] = out_m
    return out


def _finalize_chunk_catalog(chunk_res: ChunkResult, keys: np.ndarray, values: np.ndarray) -> Dict[str, np.ndarray]:
    cat = chunk_res.catalog_local
    out = {k: v for k, v in cat.items() if k != "gid"}  # includes max_sn already
    out["id"] = _map_ids_with_mapping(cat["gid"], keys, values)
    return out


# =========================
# Public API
# =========================
def fof_boolean_zarr_hpc_with_catalog(arr_bool,
                                      flux,
                                      sn,
                                      linking_length: float,
                                      *,
                                      shell: Optional[int] = None,
                                      kdtree_workers: int = -1,
                                      output_chunks: Optional[Tuple[int, int, int]] = None):
    darr = da.asarray(arr_bool)
    if darr.ndim != 3:
        raise ValueError("Expected a 3D array")

    dflux = da.asarray(flux)
    dsn = da.asarray(sn)

    # align chunking lazily
    if dflux.chunks != darr.chunks:
        dflux = dflux.rechunk(darr.chunks)
    if dsn.chunks != darr.chunks:
        dsn = dsn.rechunk(darr.chunks)

    if shell is None:
        shell = int(math.ceil(float(linking_length)))
    if shell < 1:
        raise ValueError("shell must be >= 1")

    chunks = darr.chunks
    numblocks = darr.numblocks

    offx = _prefix_offsets(chunks[0])
    offy = _prefix_offsets(chunks[1])
    offz = _prefix_offsets(chunks[2])

    delayed_blocks_bool = darr.to_delayed().ravel()
    delayed_blocks_flux = dflux.to_delayed().ravel()
    delayed_blocks_sn   = dsn.to_delayed().ravel()

    block_ids = list(product(range(numblocks[0]), range(numblocks[1]), range(numblocks[2])))

    # Stage A
    chunk_results = []
    for idx, bid in enumerate(block_ids):
        origin = _chunk_origin_from_offsets(offx, offy, offz, bid)
        cid = _linear_chunk_id(bid, numblocks)
        chunk_results.append(
            dask.delayed(_process_one_chunk)(
                delayed_blocks_bool[idx],
                delayed_blocks_flux[idx],
                delayed_blocks_sn[idx],
                origin=origin,
                chunk_id=cid,
                linking_length=linking_length,
                shell=shell,
                kdtree_workers=kdtree_workers
            )
        )

    # Stage B: cross-chunk edges
    def in_bounds(b):
        return (0 <= b[0] < numblocks[0]) and (0 <= b[1] < numblocks[1]) and (0 <= b[2] < numblocks[2])

    idx_of = {bid: i for i, bid in enumerate(block_ids)}
    neighbor_dirs = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    dir_faces = {
        (1, 0, 0): ("x1", "x0"),
        (0, 1, 0): ("y1", "y0"),
        (0, 0, 1): ("z1", "z0"),
    }

    edge_tasks = []
    for bid in block_ids:
        for d in neighbor_dirs:
            nb = (bid[0] + d[0], bid[1] + d[1], bid[2] + d[2])
            if not in_bounds(nb):
                continue
            ia = idx_of[bid]
            ib = idx_of[nb]
            fa, fb = dir_faces[d]

            edge_tasks.append(
                dask.delayed(lambda ra, rb, fa=fa, fb=fb:
                             _edges_between_faces(ra.faces[fa], rb.faces[fb],
                                                 linking_length=linking_length,
                                                 kdtree_workers=kdtree_workers)
                             )(chunk_results[ia], chunk_results[ib])
            )

    all_edges = dask.delayed(_concat_edges)(*edge_tasks)
    mapping_kv = dask.delayed(_build_mapping_from_edges)(all_edges)  # (keys, values)

    # Stage C: relabel chunks into a global label array
    relabeled_blocks = []
    for idx, bid in enumerate(block_ids):
        sx = chunks[0][bid[0]]
        sy = chunks[1][bid[1]]
        sz = chunks[2][bid[2]]
        shape = (sx, sy, sz)

        relabeled_blocks.append(
            da.from_delayed(
                dask.delayed(_relabel_chunk)(chunk_results[idx], mapping_kv[0], mapping_kv[1]),
                shape=shape,
                dtype=np.int64
            )
        )

    grid_blocks = [[[None for _ in range(numblocks[2])]
                    for _ in range(numblocks[1])]
                   for _ in range(numblocks[0])]

    for bid in block_ids:
        grid_blocks[bid[0]][bid[1]][bid[2]] = relabeled_blocks[idx_of[bid]]

    labels = da.block(grid_blocks)

    if output_chunks is not None:
        labels = labels.rechunk(output_chunks)

    # Catalog: finalize ids then merge across chunks
    chunk_cats_final = [
        dask.delayed(_finalize_chunk_catalog)(cr, mapping_kv[0], mapping_kv[1])
        for cr in chunk_results
    ]
    catalog = dask.delayed(_merge_catalog_dicts)(chunk_cats_final)

    return labels, catalog


# -------------------------
# Example usage:
# labels, cat_delayed = fof_boolean_zarr_hpc_with_catalog(arr_bool, flux_cube, sn_cube, linking_length=1.5)
# cat = dask.compute(cat_delayed)[0]
# from astropy.table import Table
# tab = Table(cat)
# -------------------------


from astropy.table import Table
from astropy.wcs import WCS


def catalog_dict_to_astropy_table(cat: dict, header=None) -> Table:
    """
    Convert FoF catalog dictionary into an Astropy Table
    and optionally convert pixel coordinates to world coordinates.

    Parameters
    ----------
    cat : dict
        Output from dask.compute(cat_delayed)[0]

    header : astropy.io.fits.Header or astropy.wcs.WCS, optional
        If provided, pixel coordinates will be converted to world
        coordinates using WCS.

    Returns
    -------
    tab : astropy.table.Table
    """
    if "id" in cat and cat["id"].size:
        order = np.argsort(cat["id"])
    else:
        order = slice(None)

    tab = Table()
    for key, val in cat.items():
        tab[key] = val[order]

    # pixel units (centroids)
    if {"x_flux", "y_flux", "z_flux"}.issubset(tab.colnames):
        tab["x_flux"].unit = "pixel"
        tab["y_flux"].unit = "pixel"
        tab["z_flux"].unit = "pixel"

    if header is not None:
        wcs = header if isinstance(header, WCS) else WCS(header)

        # numpy index order = (z,y,x) but WCS expects (x,y,z)
        if {"x_flux", "y_flux", "z_flux"}.issubset(tab.colnames):
            x = np.asarray(tab["x_flux"], dtype=float)
            y = np.asarray(tab["y_flux"], dtype=float)
            z = np.asarray(tab["z_flux"], dtype=float)
            world = wcs.all_pix2world(x, y, z, 0)
            tab["world_x"] = world[0]
            tab["world_y"] = world[1]
            tab["world_z"] = world[2]

        # Optional: extents
        extent_fields = [
            ("xmin", "ymin", "zmin", "world_xmin", "world_ymin", "world_zmin"),
            ("xmax", "ymax", "zmax", "world_xmax", "world_ymax", "world_zmax"),
        ]
        for xf, yf, zf, wx, wy, wz in extent_fields:
            if {xf, yf, zf}.issubset(tab.colnames):
                world = wcs.all_pix2world(
                    np.asarray(tab[xf], float),
                    np.asarray(tab[yf], float),
                    np.asarray(tab[zf], float),
                    0
                )
                tab[wx] = world[0]
                tab[wy] = world[1]
                tab[wz] = world[2]

    return tab


#############################

def run_fof_and_catalog(conf, incube_filename, outcube_filename, outcatalog_filename, linking_length):
    import zarr
    
    header = create_3D_header(conf)
    
    #in_path = conf.stack_thresheld_filtered_sig_name()
    print(f"Using thresheld cube {incube_filename}.")
    thresheld_sn_cube = zarr.open_array(incube_filename, mode="r")
    
    #out_path = conf.stack_segmap_name()
    #cat_out_path = conf.stack_catalog_fits_name()

    arr        = thresheld_sn_cube
    flux_cube  = zarr.open_array(conf.stack_cube_name(), mode="r")
    sn_cube    = zarr.open_array(conf.stack_sig_cube_name(), mode="r")
    
    labels, cat_delayed = fof_boolean_zarr_hpc_with_catalog(
        arr_bool=arr,          # boolean mask cube for FoF
        flux=flux_cube,        # float cube
        sn=sn_cube,            # float cube (currently unused in catalog, but wired in)
        linking_length=linking_length,
        output_chunks=(64, -1, -1),
    )
    
    labels_z = labels  # dask array
    
    cat = dask.compute(cat_delayed)[0]  # dict of numpy arrays
    
    # Optional: convert to Astropy Table
    tab = Table(cat)
    
    labels.to_zarr(outcube_filename, overwrite=True)
    print(f"Wrote {outcube_filename}.")
    
    #h = stack.create_3D_header(conf)
    tab =  catalog_dict_to_astropy_table(cat, header=header)   


    #from astropy.table import Table
    Table( tab ).write( outcatalog_filename , overwrite=True)
    print(f"Wrote {outcatalog_filename}.")
    


##### Catalog augmentation ##############


