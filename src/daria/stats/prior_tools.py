import os
import numpy as np

norm_sfr_priors = {
    'norm_sfr_0': (-5,-1), # log10
    'norm_sfr_a': (-3,3),
    'norm_sfr_z': (-2,0),
    'norm_sfr_logz': (-1,1)
    }

mbreak_sfr_priors = {
    'mbreak_sfr_0': (10,13), # log10
    'mbreak_sfr_z': (-1,1)
    }

slope_sfr_priors = {
    'slope_lo_sfr': (0,4),
    'slope_hi_sfr': (0,4)
    }

mmin_priors = {
    'mmin_0': (6,10), # log10
    'mmin_z': (-1,1)
    }

dust_priors = {
    'norm_Av_0': (0,3),
    'norm_Av_z': (-0.5,0.5),
    'slope_Av': (0,0.5)
    }

metal_ip_priors = {
    'N2_norm': (-1,0),
    'N2_slope': (0,0.5),
    'O3_norm': (-2,2),
    'O3_slope': (-1,1)
    }

total_priors = [mmin_priors,norm_sfr_priors,mbreak_sfr_priors,\
                slope_sfr_priors,dust_priors,metal_ip_priors]

def assemble_priors(include_params):
    # assemble dict of default priors as defined above
    priors_dict = {}
    
    # assemble dict of ALL default priors
    for mini_dict in total_priors:
        priors_dict.update(mini_dict)

    # remove parameters not in include_params
    tot_keys = list(priors_dict.keys())
    for key in tot_keys:
        if key not in include_params:
            priors_dict.pop(key)

    return priors_dict

def get_priors_label(priors):
    # assemble unique label corresponding to priors
    label = ''
    for i, (k,v) in enumerate(priors.items()):
        temp = f'{k}_{v[0]}_to+{v[1]}'
        label = temp if i == 0 else os.path.join(label,temp)
    return label

def get_pos(priors,nwalkers):
    # get random initial positions for MCMC walkers
    params = list(priors.keys())
    pos = np.zeros((nwalkers,len(params)))

    # draw random initial positions within the prior volume
    for i, (param, prior) in enumerate(priors.items()):
        pos[:,i] = np.random.uniform(low=prior[0],high=prior[1],\
                                     size=nwalkers)

    # for the DPL slopes, sort to enforce lo < hi
    if ('slope_lo_sfr' and 'slope_hi_sfr') in params:
        i_lo = params.index('slope_lo_sfr')
        i_hi = params.index('slope_hi_sfr')
        pos[:,[i_lo,i_hi]] = np.sort(pos[:,[i_lo,i_hi]],axis=1)

    return pos
