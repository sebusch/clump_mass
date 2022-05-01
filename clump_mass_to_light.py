from __future__ import division
import matplotlib #as mpmpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import warnings
from copy import deepcopy
from astropy.io import fits
from astropy.modeling import models, fitting
import astropy.wcs
from astropy import units as u
from astropy.nddata import CCDData
from astropy import log
from astropy.visualization import (imshow_norm, MinMaxInterval,PercentileInterval,
                                   SqrtStretch,LogStretch)
from scipy.optimize import leastsq
from scipy.interpolate import interp1d
import pandas as pd
import subprocess

# shouldn't need to change this
filter_extinction_ratio = {
    'F225W': 2.67,
    'F336W': 1.69,
    'F467M': 1.22,
    'FR647M': -999, #This will be calculated for individual target
    'F125W': 0.31
}

# center wavelengths of FR647M for all the objects
#wavelengths = [6454,6403,6783,6797,6798,6852,6460]

def load_image_set(container, params, set_wcs_to_reference = False):
    '''
    # use set_wcs_to_reference = True if images are aligned in pixels but the wcs is still offset.
    # This will set all wcs in 'raw' to be equal to the wcs_reference_filter.
    # Defaults to False, where the original wcs is kept for each image.
    
    fills in the container keys:
        entries that contain the set of filters:
            'raw','pix_to_mJy'
        stand-alone entries for whole set:
            'arcsec_per_pix', 'gal_coords', 'filters'
    sets global variables gal_coords and clump_skycoords if container is the "reference_image_set"        
    
    '''
    gal_size = params['gal_size']
    gal_coords = params['gal_coords']
    clump_skycoords = params['clump_skycoords']
    wcs_reference_filter = params['wcs_reference_filter']
    reference_image_set = params['reference_image_set']
        
    container['filters'] = list(container['filenames'].keys())
    
    for filt in container['filters']:    
        # add any missing dictionary keys:
        for key in ['raw','pix_to_mJy']:
            if key not in container:
                container[key] = {}
        container['raw'][filt] = CCDData.read(container['filenames'][filt], unit = "electron / s")
        container['pix_to_mJy'][filt] = pix_val_to_mJy(container['raw'][filt].header, params)

    wcs_reference_image = container['raw'][wcs_reference_filter]
    
    if set_wcs_to_reference:
        for filt in container['filters']:
            container['raw'][filt].wcs = wcs_reference_image.wcs
    
    container['arcsec_per_pix'] = astropy.wcs.utils.proj_plane_pixel_scales(wcs_reference_image.wcs)[0]*(u.deg).to(u.arcsec)
    
    if container == reference_image_set:
        # sets global variables
        if not isinstance(gal_coords, astropy.coordinates.SkyCoord):
            # tests if gal_coords is already set with a SkyCoord object.
            params['gal_coords'] = astropy.coordinates.SkyCoord(
                                    wcs_reference_image.header['RA_TARG'],
                                    wcs_reference_image.header['DEC_TARG'], unit="deg")
        params['clump_skycoords'] = get_clump_skycoords(container, params)
        
###############################################
###############################################
#### extract header information for calibration
#### 


def pix_val_to_mJy(header, params):
    return 1000*inverse_sensitivity(header)*extinction_correction(header, params)

def inverse_sensitivity(header):
    '''
    reads header for keywords
    returns inverse sensitivity, Jy*sec/e-
    
    If PHOTFNU keyword not available, calculates from
    PHOTFLAM/(2.9979246*10^-5 * PHOTPLAM^-2)
    '''
    
    if 'PHOTFNU' in header.keys():
        #print('photfnu')
        return header['PHOTFNU']
    else:
        #print('photflam')
        return header['PHOTFLAM']/(2.9979246*10**-5*(header['PHOTPLAM'])**-2)
# Option to use astropy.units to do the conversion, wth these 2 lines below.
# I tested the above line and the below and they are equivalent.           
#         temp = image_header['PHOTFLAM']*(u.erg/u.cm**2/u.s/u.AA)
#         return temp.to(u.Jy, equivalencies = u.spectral_density(image_header['PHOTPLAM']*u.AA)).value
        
def extinction_correction(header, params):
    '''
    reads header for filter name
    if filter_extinction_ratio < 0 (flagged -999 in definitions)
    '''
    global filter_extinction_ratio
    Av = params['Av']
    
    filter_name = get_filter(header)
    if filter_name != 'FR647M':
        extinction_ratio = filter_extinction_ratio.get(filter_name)
    else:
        wavelength = header['PHOTPLAM']
        extinction_ratio = calc_A_ratio(wavelength)

    if extinction_ratio < 0:
        raise ValueError("Invalid extinction ratio for {}".format(filter_name))
    return 10**(0.4*Av*extinction_ratio)

def get_filter(header):
    if 'FILTER' in header.keys():
        return header['FILTER']
    elif 'FILTER2' in header.keys():
        return header['FILTER2']
    else:
        return header['PHOTMODE']
            

def calc_A_ratio(wavelength):
    '''
    wavelength: in Angstroms.
    
    Returns ratio between extintion at input wavelength and visual extintion. (Aλ/AV)
    
    Calculated using the extinction law by Fitzpatrick (1999) 
    improved by Indebetouw et al (2005) in the infrared. 
    You can download the corresponding data file for the opacity kλ. 
    (http://svo2.cab.inta-csic.es/theory/fps3/getextlaw.php)
    The extinction at each wavelength is calculated as: 
    Aλ/AV = kλ/kV, 
    where kλ is the opacity for a given λ and kV=211.4
        
    Taken from: 
    http://svo2.cab.inta-csic.es/theory/fps3/index.php?id=HST/ACS_WFC.FR647M&&mode=browse&gname=HST&gname2=ACS_WFC#filter
    clicked on ? next to Af/AV
    '''
    
    kV=211.4
    
    # fitzIndeb contains relevant wavelengths for the fr647m region copied from data file
    # format: wave(A)   opacity(cm2/g)
    fitzIndeb= '''
6.00000e+3 192.366
6.10000e+3 188.477
6.20000e+3 184.609
6.30000e+3 180.794
6.40000e+3 176.987
6.50000e+3 173.134
6.60000e+3 169.218
6.70000e+3 165.328
6.80000e+3 161.572
6.90000e+3 158.053
7.00000e+3 154.796
'''
    # these 2 lines process the fitzIndeb string above into a 2D array [wavelength array, opacity array]
    l = [line.split(' ') for line in fitzIndeb.strip().split('\n')]
    fitzIndeb_arr = np.array(l,dtype='float').T
    
    # create linear interpolate function
    # index 0: array of wavelength
    # index 1: array of opacity
    f = interp1d(fitzIndeb_arr[0], fitzIndeb_arr[1], kind='linear')
    
    # f(wavelength) is the interpolated value of opacity at wavelength (kλ)
    # returns (kλ/kV) rounded to 2 decimal points.
    return round(f(wavelength)/kV,2)

##############################################
       
        
def set_npix(container, clump_size):
    # clump_size is the clump diameter in arcsec
    # sets the npix parameter to be total size of the box in pixels of the clump (odd number)
    container['npix'] = 1+2*round((0.5*clump_size/container['arcsec_per_pix']).value)
    container['radius_image_npix'] = create_radius_image(container['npix'])
    

def create_radius_image(shape):
    if isinstance(shape,int):
        shape = (shape,shape)
    radius_image = np.zeros(shape=shape)

    center_x = int(round(0.5*(shape[0]-1)))
    center_y = int(round(0.5*(shape[1]-1)))

    x,y = np.indices(shape)
    x-=center_x
    y-=center_y
    radius_image = np.sqrt(x*x + y*y  )
    return radius_image


def compute_sky_stats(container, params, num_per_side, num_clip):
    '''
    creates 2d cutout of sky_size from each filter in container['raw']
    retrieves npix from container (this is clump size)
    calls get_edge_patches(cutout, npix, num_per_side, num_clip) for each filter
    saves result to container in container['sky_stats'], container['sky_stats_mJy'], and container['sky_patches']
    '''
    sky_size = params['sky_size']
    gal_coords = params['gal_coords']
#     wcs = container['wcs']
    npix = container['npix']
    
    container['sky_stats'] = {}
    container['sky_stats_mJy'] = {}    
    container['sky_patches'] = {}
    
    for filt in container['filters']:
        
        this_image = container['raw'][filt]
        cutout = astropy.nddata.Cutout2D(this_image,gal_coords,sky_size,wcs=this_image.wcs)
        patches, med_std = get_edge_patches(cutout.data, npix, num_per_side, num_clip)
        container['sky_stats'][filt] = med_std
        container['sky_stats_mJy'][filt] = np.array(med_std)*container['pix_to_mJy'][filt]
        container['sky_patches'][filt] = patches
        
def get_edge_patches(image,npix,num_per_side, num_clip):
    '''
    Retrieves 4*num_per_side - num_clip patches of npix x npix around edge of image
    (removes the num_clip patches with highest median value)
    outpus: [list of patches], median of patch median, median of patch standard deviation
    '''
    patches = []

    tot_size = min(image.shape)
    indicies = (np.arange(num_per_side)*tot_size/num_per_side).astype(int)
    
    for i in indicies:
        patches.append(image[0:npix,i:i+npix])
        patches.append(image[tot_size-(i+npix):tot_size-i,0:npix])
        patches.append(image[-npix:,tot_size-(i+npix):tot_size-i])       
        patches.append(image[i:(i+npix),-npix:])
        
    patch_stats = []
    for patch in patches:
        patch_stats.append([np.median(patch), np.std(patch) ,patch])
    if num_clip > 0:    
        patch_stats = np.array(sorted(patch_stats)[:-num_clip])
    else:
        patch_stats = np.array(patch_stats)
    patches = np.array([patch[2] for patch in patch_stats])
    return patches, [np.median(patch_stats[:,0]), np.median(patch_stats[:,1])]

def plot_sky_patch_hists(container):
    hist_bins=50
    for filt in container['filters']:
        patches = container['sky_patches'][filt]
        median, sigma = container['sky_stats'][filt]
        xrange=(median-10*sigma,median+50*sigma)
        fig = plt.figure()
        plt.hist(container['raw'][filt].data.flatten(), bins=hist_bins,log=True, 
                    range=xrange, color='lightgrey', label="sky region")
        plt.gca().set_ylabel("sky region count")
        ax2 = plt.gca().twinx()
        ax2.hist([patch.flatten() for patch in patches],histtype='step',log=True,range=xrange, bins=hist_bins)
        # for patch in patches:
        #     ax2.hist(patch.flatten(),histtype='step',color='black',log=True,range=xrange, bins=hist_bins)
        plt.gca().set_title("{} {}, med, std = {:.3g}, {:.3g}".format(container['name'], filt, median,sigma))
        plt.axvspan(xmin = median - sigma, xmax = median+sigma, color='blue', alpha=0.2, label='stdev')
        plt.axvline(x=median, color='red', label='median')
        ax2.set_ylabel("patch region count")
        ax2.legend()     

def sky_subtract_gal_region_mJy(container, params):
    '''
    creates the galaxy region cutout container['gal_region_raw'] (units: electrons/s)
    for each filt in container['gal_region_raw']:
    subtract sky median (in units of e-/s)
    multiply result by scaling of e-/s to mJy (value is stored in container['pix_to_mJy'])
    store this in: container['gal'][filt] (units: mJy)
    '''
    gal_coords = params['gal_coords']
    gal_size = params['gal_size']

    sky_stats = container['sky_stats']
    
    container['gal'] = {}
    container['gal_region_raw'] = {}

    for filt in container['filters']:
        # extract cutout2D for raw galaxy:
        raw_image = container['raw'][filt]
        gal_patch = astropy.nddata.Cutout2D(raw_image,gal_coords, gal_size, wcs=raw_image.wcs)
        container['gal_region_raw'][filt] = gal_patch
        gal_patch_mJy = deepcopy(gal_patch)

        sky_med = sky_stats[filt][0]
        scale = container['pix_to_mJy'][filt]

        # scale = 1 #for debug purposes
        if scale ==1:
            log.warning('NOT IN mJy! Conversion e-/s to mJy has been manuall set '
                    'to 1 compare raw flux to other code')

        gal_patch_mJy.data = scale * (gal_patch.data - sky_med)
        container['gal'][filt] = gal_patch_mJy
#        container['wcs_gal'] = gal_patch.wcs
    
def get_clump_skycoords(container,params):
    '''
    read container['clumpx'] and container['clumpy'] (1-based pixel position) 
    create SkyCoord based on container['raw'][wcs_reference_filter].wcs
    '''
    wcs_reference_filter = params['wcs_reference_filter']

    wcs = container['raw'][wcs_reference_filter].wcs
    clumpx = container['clumpx']
    clumpy = container['clumpy']
    return astropy.wcs.utils.pixel_to_skycoord(clumpx,clumpy,wcs,origin=1)

def make_skymask(container, sigma_factor=3):
    '''
    the units of all parts of this calculation is intended to be mJy.
    computes a mask where 1 = any filter have a value < sigma_factor * sky_sigma (all units mJy)
    assigns to container['skymask']
    '''
    
    sky_stats = container['sky_stats_mJy'] # each filt contains array of [median, sigma], in mJy
    mask_array = [] # storage for results of each filter test
    for filt in container['filters']: # the gal image has already been scaled to mJy
        mask_array.append(container['gal'][filt].data < sigma_factor * sky_stats[filt][1])
    container['skymask'] = np.any(mask_array, axis=0)

    
def print_sky_stats(container):
    print(container['name'], "sky stats [med, stdev]")
    for filt in container['filters']:
        print(' {} raw: [{:.4g}, {:.4g}]    mJy: [{:.4g}, {:.4g}]'.format(filt,*container['sky_stats'][filt],*container['sky_stats_mJy'][filt]))
        
def get_clump_cutouts(container, params):
    clump_skycoords = params['clump_skycoords']
    npix = container['npix'] # extract npix x npix region
    gal = container['gal'] # extract from gal images
    
    container['clumps'] = {}
    for i in range(len(clump_skycoords)):
        clump_name = i+1
        container['clumps'][clump_name] = {}
        for filt in container['filters']:
            container['clumps'][clump_name][filt] = {}
            this_clump = container['clumps'][clump_name][filt] 

            image = gal[filt]
            this_clump['orig'] = astropy.nddata.Cutout2D(
                image.data,clump_skycoords[i],npix,wcs=image.wcs)


############################################################
############################################################
### adding in the clump fit code
############################################################
############################################################

#### ALL ALREADY IN mJy
#clump params: num_passes, disk_subtract_flag
def clumpSmooth(container, clumpNum, filter_key, clump_params):
    num_passes = clump_params['num_passes']
    disk_subtract_flag = clump_params['disk_subtract_flag']
    radius_image = container['radius_image_npix']

    this_clump = container['clumps'][clumpNum][filter_key]
    # add any missing dictionary keys:
    for key in ['smooth','radial','flags']:
        if key not in this_clump:
            this_clump[key] = {}

    clump_cutout = this_clump['orig']
    clump = clump_cutout.data # retrieves the underlying data.

    clump_smooth_cutout = deepcopy(clump_cutout)
    clump_smooth = clump_smooth_cutout.data # retrieves the underlying data.
    this_clump['smooth'] = clump_smooth_cutout

    for i in range(num_passes-1):
        ### calculate median and stdev vs radius
        this_clump['radial'] = {}
        # returns radius_profile, median, stdev, fit_clump
        this_clump['radial'] = radial_profile(container, clump_smooth)
        #replace_aximuthal modifies the input image, clump_smooth
        #and returns a boolean image showing the failed pixels.
        #this first pass we will set the failed pixels to nan
        this_clump['flags'] = replace_azimuthal(container, this_clump, 
                    filter_key, flag_conditions, replace_value=float('nan'))
   
    # returns radius_profile, median, stdev, fit_clump
    this_clump['radial'] = radial_profile(container, clump_smooth)
    this_clump['flags'] = replace_azimuthal(container, this_clump, 
                filter_key, flag_conditions, replace_value="default")


    # part 3. Define size of clump.
    # set values outside of 3*sigma radius = nan
    fit_clump = this_clump['radial']['gauss_fit']
    if fit_clump[1] > 0:
        clump_smooth[radius_image > 2.0*fit_clump[1]] = float('nan')
        #clump_smooth[radius_image >1.28] = float('nan')


    # part 4. Disk subtraction
    ###
    ### Disk subtraction: Below I subtract the baseline, taken from the gaussian fit parameters
    ### Note, this modifies clump_smooth and median_r

    disk_value = 0 #initialize value.
    if disk_subtract_flag is True:
        disk_value = fit_clump[2]
        clump_smooth -= disk_value
        this_clump['radial']['median'] -= disk_value


# for the input image (parameter: clump),
# This function calculates the radial profile of the image.
# it oversamples the radius (radius increments by 0.5 pixel)
# returns:
#         radius_profile = monotonically increasing radius value
#         median = median of all pixels within 1 pix of radius r, where r is radius_profile at the same index
#         stdev = standard deviation of all pixels within 1 pix of radius r, where r is radius_profile at the same index
#         fit_clump[0] = the gaussian fit parameters (amplitude, sigma, y offset)
#         np.array(radius_image).reshape(clump.shape) = image containing value = radius from center pix.

def radial_profile(container, clump):
    # these 2 1d arrays have related indexing
    radius_image = container['radius_image_npix']
    clump_list = clump.flatten()

    # create equally spaced radius list, has sampling of 1 pixels.
    # these 3 arrays have related indexing
    radius_profile = np.arange(0, np.max(radius_image), 0.5)
    median = np.array([])
    stdev = np.array([])

    # loop over image and test if the radius of the image pixel is near
    # enough to the selected radius in radius_profile.
    # find median and standard deviation for all the selected pixels.
    for l in range(len(radius_profile)):
        this_radius = radius_profile[l]

        # array of indicies where condition is true.
        temp_index = np.where(
                                (radius_image >= this_radius - 0.5)
                                &
                                (radius_image <= this_radius + 0.5)
                            )
        if len(temp_index[0]) > 0:
            median = np.append(median, np.nanmedian(clump[temp_index]))
            stdev = np.append(stdev, np.nanstd(clump[temp_index]))
        else:
            median = np.append(median, median[l-1])
            stdev = np.append(stdev, stdev[l-1])

    # fit gaussian to median profile:
    # estimate offset, sigma as initial guess into model
    offset_0 = np.min(median)
    sigma_0 = 1.5
    amplitude_0 = median[0] - offset_0

    # model is gauss function defined globally,
    # with input x-array, coefficients (c1, sigma1, offset)
    def gauss_residuals(coeffs):
        # uses x-array radius_profile
        # uses y-array median
        model = gauss(radius_profile, coeffs)
        return median - model

    # do the fit
    fit_clump = leastsq(gauss_residuals, [amplitude_0, sigma_0, offset_0])
    # print(fit_clump)
    results = {}
    results['r'] = radius_profile
    results['median'] = median
    results['stdev'] = stdev
    results['gauss_fit'] = tuple(fit_clump[0])
    return results
    # return {
    #     'r':radius_profile, 
    #     'median':median, 
    #     'stdev':stdev, 
    #     'gauss_fit':fit_clump[0]
    # }
    
    # plt.plot(radius_image.flatten(), clump.flatten(), '.')
    # plt.errorbar(x=radius_profile, y=median, yerr=stdev, fmt='k' )

#### Gaussian profile fitting (assumes peak at x=0)
def gauss(x,pars):
    (c1, sigma1, offset) = pars

    gs =   offset + c1 * np.exp( - (x)**2.0 / (2.0 * sigma1**2.0))
    return gs

# function that replaces pixels in image
# Compares each pixel to it's corresponding median and standard deviation at that radius.
# Conduct a series of tests on the pixel value
# If pixel satisfies any condition, replace it with replace_value (defaults to the median for that radius)
# replace_value may be omitted from function call to use the median_r list.
# if replace_value is used, it must be a single value (e.g. float('nan'), or -999)
#
# returns a boolean array showing which pixels were replaced.
def replace_azimuthal(container, this_clump, filt, tests, replace_value="default"):
    image = this_clump['smooth'].data
    npix = int(round((container['npix'] - 1)/2)) # converts diameter to radius
    # container['npix'] is diameter. for this function's npix: image[npix,npix] should be center
    
    radius_list = this_clump['radial']['r']
    median_r = this_clump['radial']['median']
    stdev_r = this_clump['radial']['stdev']

    value_flags = np.full(image.shape, False, dtype=bool)
    for m in range(len(image)):
        for n in range(len(image[m])):
            # using clump_radial[0] as the radius array:
            # find the index of radius closest to the radius of the pixel,
            # assign it to the variable called position
            # use this index for appropriate median, stdev selection for that pixel.
            rad_diff = 100 #start with large unphysical value
            for l in range(len(radius_list)):
                temp_rad_diff = abs(radius_list[l] - np.sqrt((m-npix)**2 + (n-npix)**2))
                if rad_diff > temp_rad_diff:
                    rad_diff = temp_rad_diff
                    position = l

            ## define the conditions which flag bad pixels
            conditions = tests(image, m,n,position, radius_list, median_r, stdev_r, npix)

            ## loop over conditions. if any condition is true,
            ## set pixel to median value
            if any(conditions):
                if replace_value=="default":
                    image[m,n] = median_r[position]
                else:
                    image[m,n] = replace_value
                value_flags[m,n]=True

    return value_flags
## define the conditions which flag bad pixels
# this function returns an array containing boolean values based on the tests.
# Pixels where any test is True will be replaced.
def flag_conditions(image, index_x, index_y, rad_index, radius_list, median_r, stdev_r, npix):
    # npix here is radius, such that image[npix, npix] is center of image.
    image_pix_value = image[index_x, index_y]
    image_center_value = image[npix,npix]
    this_radius = radius_list[rad_index]
    median = median_r[rad_index]
    stdev = stdev_r[rad_index]

    return [
            image_pix_value > median + 4.*stdev
            and
            this_radius > 4.,

            image_pix_value > median + 3.*stdev
            and
            this_radius > 3.0,

            image_pix_value  > median + 2.0*stdev
            and
            this_radius > 1.,

            # This next two if statements makes the assumption that bright pixels that are off center are
            # not associated with this clump and are removed from the fit. center = [npix,npix]
            image_pix_value  > 0.75*image_center_value
            and
            this_radius > 5.0,

            image_pix_value  > image_center_value
            and
            this_radius > 3.,

            np.isnan(image_pix_value)
        ]



##########################
##########################
### ploltting
##########################
##########################

def plot_r_profile(container, clumpNum, filter_label, clump_params, ax=None):
    this_clump = container['clumps'][clumpNum][filter_label]
    clump = this_clump['orig'].data
    clump_smooth = this_clump['smooth'].data
    clump_r_profile = this_clump['radial']
    disk_subtract_flag = clump_params['disk_subtract_flag']

    radius_image = container['radius_image_npix']

    radius_list = clump_r_profile['r']
    median_r = clump_r_profile['median']
    stdev_r = clump_r_profile['stdev']
    fit_clump = clump_r_profile['gauss_fit']
    disk_value = fit_clump[2]

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()
    ax.set_title("clump {}, filter {}".format(clumpNum, filter_label))
    ax.set_title("c = {:.3g}, sigma = {:.3g}, y0 = {:.3g}".format(*fit_clump))
    ax.plot(radius_list, median_r, '.-k', label='radial median')
    ax.errorbar(x=radius_list, y=median_r, yerr=stdev_r, fmt='k')
    ax.plot(radius_image.flatten(), clump.flatten(), '.',label='original')
    ax.plot(radius_image.flatten(), clump_smooth.flatten(), 'cx',label='smoothed')
    ax.plot(radius_list,-(disk_subtract_flag)*disk_value + gauss(radius_list,fit_clump),
            'r', label='gauss fit',linewidth=3)
    ax.legend(loc=2,bbox_to_anchor=(1,1))
    #ax.set_xlim([-0.1,pixwide*1.5])
    ax.set_xlabel('radius (pixels)')
    ax.set_ylabel('Flux (mJy)')
    plt.tight_layout()
    return ax

def plot_clump_panel(container, clumpNum, filter_label, clump_params):
    vmin = None#clump_params.get('vmin',None)
    vmax = None#clump_params.get('vmax',None)
    this_clump = container['clumps'][clumpNum][filter_label]

    clump = this_clump['orig'].data
    clump_smooth = this_clump['smooth'].data
    clump_flags = this_clump['flags'].data

    disk_subtract_flag = clump_params['disk_subtract_flag']
    skystats = container['sky_stats_mJy']
    # skystats = container['sky_stats'] # for debugging
    if skystats[filter_label][0] == container['sky_stats'][filter_label][0]:
        log.warning('The skystats are in e-/s. Be sure to change back to mJy after debugging')
    
    disk_value = this_clump['radial']['gauss_fit'][2]
    median_r0 = this_clump['radial']['median'][0]
    s2n = median_r0/skystats[filter_label][1]

    fig, ax_array = plt.subplots(1,4,figsize=(12,3))

    ax = ax_array[3]
    plot_r_profile(container, clumpNum, filter_label, clump_params,ax=ax)
    auto_min, auto_max = ax.set_ylim() # gets the full extent of y axis.

    if vmin is None:
        vmin = auto_min
    if vmax is None:
        vmax = auto_max
    ax.set_ylim([vmin, vmax])


    ax = ax_array[1]
    ax.set_title("smoothed")
    ax.imshow(clump_smooth, cmap='gist_yarg', interpolation='none',origin='lower',vmin=vmin,vmax=vmax)
    ax.text(0,1, r'Flagged: Flux =' '%.3f' % np.nansum(clump_smooth), fontsize=11, color='blue')
    ax.text(0,2, r's/n =' '%.3f' % s2n, fontsize=11, color='blue')
    ax.set_xlabel('{:.2%} flux replaced'.format(
        np.nansum(clump_smooth[clump_flags])/np.nansum(clump_smooth)))

    ax = ax_array[2]
    ax.set_title("map of replaced pixels")
    ax.imshow(clump_flags, cmap='gist_yarg', interpolation='none',origin='lower')

    if disk_subtract_flag is True: # if true, add back in the disk_value to the limits so that the brightness is the same as the other plot
        vmin += disk_value
        vmax += disk_value

    ax = ax_array[0]
    #print(this_clump)
    ax.set_title("clump {}, filter {}. Original".format(clumpNum, filter_label))
    ax.imshow(clump, cmap='gist_yarg', interpolation='none',origin='lower',vmin=vmin,vmax=vmax)
    ax.text(0,1, r'Raw: Flux =' '%.3f' % np.nansum(clump), fontsize=11, color='blue')
    ax.text(0,-1, r'Clump Number:' '%i' % clumpNum , fontsize=11, color='blue')
    ax.set_xlabel('pixels')
    ax.set_ylabel('pixels')



def myprint(d, print_prefix_rec=""):
    print_prefix="  "
    for k, v in d.items():
        if isinstance(v, dict):
            print("{}{}:".format(print_prefix_rec,k))
            myprint(v, print_prefix_rec+print_prefix)
        else:
            if isinstance(v, np.ndarray):
                print("{}{}: numpy.ndarray shape: {}, containing {}".format(print_prefix_rec,k, v.shape,v.dtype))
            elif isinstance(v, list):
                print("{}{}: list length: {}, containing {}".format(print_prefix_rec,k, len(v),type(v[0])))
            else:
                print("{}{}: {}".format(print_prefix_rec,k, type(v)))
        if isinstance(k, int):
            print("{} repeats for total of {} times".format(print_prefix_rec, len(d.keys())))
            break

def save_gal_mJy_skymasked(container, filename, params):
    filt_names = params['cigale_filter_names']
    redshift = params['redshift']
    skymask = container['skymask']
    gal_mJy = container['gal']
    sky_stats = container['sky_stats_mJy']
    output_fmt = '%.10g'


    valid_idx = np.where(~skymask)
    position = np.empty_like(skymask, dtype=object)

    for i in range(len(valid_idx[0])):
        row = valid_idx[0][i]
        col = valid_idx[1][i]
        position[row,col]="{}-{}".format(row,col)
    output_arr = [np.hstack(('full_gal',position[valid_idx])), 
            np.hstack((redshift, np.full_like(position[valid_idx],redshift)))]
    header = ['id','redshift']
    fmt = ['%s',output_fmt]
    for filt in container['filters']:
        filt_data = gal_mJy[filt].data
        filt_sky_sigma = sky_stats[filt][1]
        full_gal = np.sum(filt_data[valid_idx])
        output_arr += [np.hstack((full_gal, filt_data[valid_idx])), 
                np.hstack((filt_sky_sigma,(np.full_like(position[valid_idx],filt_sky_sigma))))]
        header += [filt_names[filt],filt_names[filt]+"_err"]
        fmt += [output_fmt]*2

    np.savetxt(filename, np.array(output_arr).T, fmt=fmt, header = '    '.join(header))


def load_cigale_results(filename, container, include_columns):

    table = astropy.table.Table.read(filename, format='ascii', include_names=['id']+include_columns)
    
    container['cigale'] = {}
    results = container['cigale']
    # take care of the row that must be treated differently:
    # store it in container and remove it from the table.
    for row in table:
        if row['id'] == 'full_gal':
            results['full_gal'] = dict(zip(list(row.columns),list(row)))
            table.remove_row(row.index)
            break
    
    # initialize an image 
    empty_image = deepcopy(container['gal'][container['filters'][0]])
    empty_image.data[:,:] = 0

    # set the skymasked points to nan
    empty_image.data[container['skymask']] = np.nan

    # for each of the columns in include_columns, set as a copy of empty image
    for col in include_columns:
        results[col] = deepcopy(empty_image)

    for row in table:
        array_idx = tuple([int(i) for i in row['id'].split('-')])
        for col in include_columns:
            results[col].data[array_idx] = row[col]


    #     for row in table:
    #         if row['id'] == 'full_gal':
    #             container['full_gal_cigale_results'] = row
    #         else:
    #             array_idx = tuple([int(i) for i in row['id'].split('-')])
    #             result.data[array_idx] = row['best.stellar.m_star']

    #     # set result to container[name]
    #     container[name] = result

def pcigale_check(folder = '.', quit_if_error=True):
    log.info('running: $ pcigale check')
    result = subprocess.run(["pcigale", "check"],cwd=folder, capture_output=True) 
    if result.stderr:
        log.error(result.stderr.decode('utf-8'))
        if quit_if_error:
            raise SystemExit("Error running $ pcigale check")
    log.info(result.stdout.decode('utf-8')) 

def pcigale_run(folder = '.', quit_if_error=True):
    var = input("Enter y to run: $ pcigale run     ")

    if str(var).lower() == 'y':

        log.info('running pcigale. This may take a while, with no visual progress indicator.')
        result = subprocess.run(["pcigale", "run"],cwd=folder, capture_output=True) 
        if result.stderr:
            log.error(result.stderr.decode('utf-8'))      
            if quit_if_error:  
                raise SystemExit("Error running $ pcigale run")
        log.info(result.stdout.decode('utf-8'))
    else:
        if quit_if_error:  
            raise SystemExit("User cancelled $ pcigale run")
        else:
            log.warning("User cancelled $ pciagle run. Continuing the script anyway.")


def copy_to_clumps(params, origin, to_container, to_name, to_filter=''):
    #to_filter = params['m2l_filter']
    for clump in to_container['clumps']:
        if to_filter in to_container['filters']:
            this_clump = to_container['clumps'][clump][to_filter]
        else:
            this_clump = to_container['clumps'][clump]
        # new image: copy of orig. will reset data.
        regrid_image = deepcopy(to_container['clumps'][clump][to_container['filters'][0]]['orig'])

        # keeping this for now... it was originally intended to copy the 
        # 'smooth' image, which has nans outside of 2sigma.

        valid_idx = np.where(~np.isnan(regrid_image.data))
        valid_skycoords = regrid_image.wcs.array_index_to_world(*valid_idx)
        origin_idx = origin.wcs.world_to_array_index(valid_skycoords)

        # convert (row_array, col_array) to array of (row, col) entries:
        valid_idx_arr = list(zip(*valid_idx))
        origin_idx_arr = list(zip(*origin_idx))

        # clear array to start with empty data.
        regrid_image.data[:,:] = np.nan
        for i in range(len(valid_skycoords)):
            try:
                regrid_image.data[valid_idx_arr[i]] = origin.data[origin_idx_arr[i]]
            except IndexError as e:
                print(e, ' continuing....')
                pass

        this_clump[to_name] = regrid_image

def create_m2l(container, params):
    filt = params['m2l_filter']
    m2l = deepcopy(container['cigale']['best.stellar.m_star'])
    l = container['gal'][filt]
    m2l.data = m2l.data/l.data
    if container.get('m2l') is None:
        container['m2l'] = {}
    container['m2l'][filt] = m2l

def compute_mass_images(container, filt):
    
    for clump in container['clumps']:
        this_clump = container['clumps'][clump][filt]
        l_orig = this_clump['orig']
        l_sm = this_clump['smooth']
        m2l = this_clump['m2l']

        m_orig = deepcopy(l_orig)
        m_orig.data = m2l.data*l_orig.data

        m_sm = deepcopy(l_sm)
        m_sm.data = m2l.data*l_sm.data

        this_clump['mass_orig'] = m_orig
        this_clump['mass_smooth'] = m_sm

def collect_results(container_fine, container_coarse, clump_params, filt):
    disk_subtract_flag = clump_params['disk_subtract_flag']
    size_filter = clump_params['clump_size_filter']
    # loop over clumps and insert various parameters into a pandas dataframe
    df = pd.DataFrame()
    df.index.name = 'clumpNum'

    container = container_fine
    radius_image = container['radius_image_npix']
    radius_image_coarse = container_coarse['radius_image_npix']

    sky_sigma = container['sky_stats_mJy'][filt][1]

    for clumpNum in container['clumps']:
        this_clump = container['clumps'][clumpNum][filt]
        gauss_params = [*deepcopy(this_clump['radial']['gauss_fit'])]
        # df.loc[clumpNum,'sigma.arcsec'] = gauss_params[1]*container['arcsec_per_pix']
        df.loc[clumpNum, 's/n peak'] = this_clump['radial']['median'][0]/sky_sigma

        # use specific filter for radius:
        radius_px = 2*container['clumps'][clumpNum][size_filter]['radial']['gauss_fit'][1]
        
        inside_clump_idx = np.where(radius_image<=radius_px)
        df.loc[clumpNum,'radius.'+size_filter+'.arcsec'] = radius_px*container['arcsec_per_pix']
        df.loc[clumpNum,'flux.orig.'+filt+'.mJy'] = np.sum(this_clump['orig'].data[inside_clump_idx])
        df.loc[clumpNum,'flux.smooth.'+filt+'.mJy'] = np.nansum(this_clump['smooth'].data)

        df.loc[clumpNum,'flux%replaced.smooth.'+filt] = 100*(
            np.nansum(this_clump['smooth'].data[np.where(this_clump['flags'])])
            /np.nansum(this_clump['smooth'].data)
        )


        baseline = deepcopy(gauss_params[2])
        if disk_subtract_flag:
            gauss_params[2] = 0
            

        gauss_image = gauss(radius_image, gauss_params)

        flux_gauss = np.sum(gauss_image[inside_clump_idx])
        df.loc[clumpNum,'flux.gauss.'+filt+'.mJy'] = flux_gauss
        df.loc[clumpNum,'flux.baseline.'+filt+'.mJy'] = baseline*len(inside_clump_idx[0])
        df.loc[clumpNum,'mass.orig.'+filt] = np.nansum(this_clump['mass_orig'].data[inside_clump_idx])
        df.loc[clumpNum,'mass.smooth.'+filt] = np.nansum(this_clump['mass_smooth'].data[inside_clump_idx])

        this_clump_coarse = container_coarse['clumps'][clumpNum][filt]
        radius_coarse_px = radius_px*container['arcsec_per_pix']/container_coarse['arcsec_per_pix']

        inside_clump_coarse_px = np.where(radius_image_coarse <= radius_coarse_px)
        sum_light_orig_coarse = np.nansum(this_clump_coarse['orig'].data[inside_clump_coarse_px])
        slices_coarse = this_clump_coarse['orig'].slices_original
        mass_clump = container_coarse['cigale']['best.stellar.m_star'].data[slices_coarse]
        sum_mass_coarse = np.nansum(mass_clump[inside_clump_coarse_px])

        av_m2l = sum_mass_coarse/sum_light_orig_coarse
        df.loc[clumpNum, "mass.<m2l>*lgauss."+filt] = flux_gauss * av_m2l
        df.loc[clumpNum, "mass.<m2l>*lbaseline."+filt] = baseline*len(inside_clump_idx[0]) * av_m2l
        df.loc[clumpNum, '<m2l>=<M>/<L>|coarse.'+filt+'.M/mJy'] = av_m2l
    
    
    container['summary'] = df

def plot_sky_region(container, filt, params):
    sky_size = params['sky_size']
    gal_coords = params['gal_coords']
    this_image = container['raw'][filt]
    cutout = astropy.nddata.Cutout2D(this_image,gal_coords,sky_size,wcs=this_image.wcs)
    
    plt.figure()
    plt.subplot(projection = this_image.wcs)
    imshow_norm(this_image,interval=PercentileInterval(92.5))
    cutout.plot_on_original(color='white')
