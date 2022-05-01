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

import clump_mass_to_light as cl
from importlib import reload
reload(cl)
import pandas as pd


###############################
##### Constant definitions ####
###############################

params = {
    'Av' : 0.204, ##Galactic extiniction of target G04-1
    'redshift': 0.1298, # Redshift of target G04-1
    'cigale_filter_names': {
        'nuv': 'WFC3_UVIS_F225W',
        'u': 'WFC3_UVIS_F336W',
        'b': 'WFC3_UVIS_F467M',
        'r': 'ACS_WFC_FR647M_G04-1', #### CHANGE THIS FOR EACH OBJECT
        'j': 'WFC3_IR_F125W'
    },
    
    'gal_coords' : 0,
                     # if 0, set during loading "reference_image_set"   
                        # to override: replace 0 with an astropy SkyCoord, for examples:
                        # astropy.coordinates.SkyCoord('00h42m30s', '+41d12m00s')
                        # astropy.coordinates.SkyCoord(63.082443, -5.9133453 , unit='deg')
    'gal_size' : 7*u.arcsec, #diameter
    'clump_size' : 1.2*u.arcsec, #diameter 
    'sky_size': 25*u.arcsec, #diameter
    
    'clump_skycoords' : 0, # set during loading of "reference_image_set"
    'wcs_reference_filter': 'u', # The filter in which the clumps were identified. This param is also used to  
                                # normalize the wcs of all the images during read-in, if that option is selected
                                # (set_wcs_to_reference = True) in cl.load_image_set()
    'm2l_filter': 'r', # filter used to compute mass to light ratio.
}
fine = {
    'filenames': {
        'nuv': 'nuv_g04_gauss_r_sreg.fits',
        'u': 'u_g04_gauss_r_sreg.fits',
        'b': 'b_g04_gauss_r_sreg.fits',
        'r': 'r_g04_1_cut_matched.fits',
    },
    # clumpx, clumpy start at index=1, and are flipped from python. these must be from another program.
    ##  FR647M  1   2   3   4   5   6   7   8   9   10  11  12  13  14
    'clumpx': [492,481,475,468,457,437,427,465,460,468,464,442,446,490],
    'clumpy': [406,410,413,420,424,417,411,404,394,392,377,402,378,397],
    # clumpx and clumpy are indexed at 1. Python is indexed at 0.
    # I may write new arrays indexed at 0 so I don't have to worry about this.
    # python addresses 2d array as [row, column]
    # which would be [y,x] in a plot.
    'name': "fine" #name used in plots and print statements
}

coarse = {
    'filenames': {
        'nuv': 'nuv_g04_gauss_j_sreg.fits',
        'u': 'u_g04_gauss_j_sreg.fits',
        'b': 'b_g04_gauss_j_sreg.fits',
        'r': 'r_g04_gauss_j_sreg_shift.fits',
        'j': 'idlh01040_drz_sreg_new.fits'
    },
    'name': "coarse" #name used in plots and print statements
}

# must be set after the corresponding variable is created
params['reference_image_set'] = fine # the name of the group that contains the 
                            # reference positions of the clumps

containers=[fine,coarse] #name of the above variables so I can loop over easy.


#################################
### load data, normalize and sky subtract, extract clump regions
#################################

for container in containers:       
    # use set_wcs_to_reference = True if images are aligned in pixels but the wcs is still offset.
    # This will set all wcs in 'raw' to be equal to the wcs_reference_filter.
    # Defaults to False, where the original wcs is kept for each image.
    cl.load_image_set(container, params, set_wcs_to_reference = True)
    cl.set_npix(container, params['clump_size']) #converts params['clump_size']/2 to (rounded up) pixels, then sets box size 2n+1 pix.

for container in containers:   
    cl.compute_sky_stats(container, params, num_per_side=4, num_clip=3) 
#     cl.print_sky_stats(container)

    cl.sky_subtract_gal_region_mJy(container, params)
    cl.make_skymask(container,sigma_factor=3)
    
    cl.get_clump_cutouts(container, params)
#    cl.plot_sky_patch_hists(container)


###############################
##### create cigale input file, run cigale, load cigale results
###############################

cl.save_gal_mJy_skymasked(coarse, 'G04-1.txt',params)
cl.pcigale_check(folder='.', quit_if_error=False)
cl.pcigale_run(folder='.', quit_if_error = False)
cl.load_cigale_results('out/results.txt',coarse,include_columns=['best.stellar.m_star','best.attenuation.V_B90'])
cl.create_m2l(coarse, params) # hardcoded to do the computation: 
            # coarse['cigale']['best.stellar.m_star'] / coarse['gal'][params['m2l_filter']]
            # outputs result to: coarse['m2l'][params['m2l_filter']]. units are solar masses/ mJy

################################
##### perform clump smoothing
################################

clump_params = {
    'num_passes': 2,
    'disk_subtract_flag': True,
    'clump_size_filter': 'u', # filter to use to set clump size.
    'vmax': None, #graph parameters
    'vmin': None, #graph parameters
}
log.setLevel('DEBUG') #options are DEBUG INFO WARNING ERROR
for container in [fine]:
    for clump in container['clumps'].keys():
        for filt in ['u','r']: # make sure that 'clump_size_filter' is in this list.
            cl.clumpSmooth(container,clump,filt,clump_params)

        
###############################
##### collect results
###############################
cl.copy_to_clumps(params, origin=coarse['m2l'][params['m2l_filter']], 
                      to_container=fine, to_name='m2l',to_filter=params['m2l_filter'])
    # the above function copies origin image into:
    # to_container['clumps'][clumpNum][to_filter][to_name]
    # or if to_filter is empty or not a valid filter, 
    # to_container['clumps'][clumpNum][to_name]
    # e.g.: coarse['m2l']['r'] --->  fine['clumps'][2]['r']['m2l']
cl.copy_to_clumps(params, origin=coarse['cigale']['best.attenuation.V_B90'], 
                      to_container=fine, to_name='best.attenuation.V_B90', to_filter=None)    
cl.compute_mass_images(fine,filt=params['m2l_filter'])
cl.collect_results(fine, coarse, clump_params, filt=params['m2l_filter'])
df = fine['summary']
print(df)

