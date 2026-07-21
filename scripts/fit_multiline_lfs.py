### imports

import os,copy,sys,time,pickle
import emcee
import numpy as np
import astropy.units as u

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm

from multiprocess import Pool, current_process

import SPHEREx_L4_EBL_Inference as spin

from daria.models import ToyGalaxyPopulation, ToyEmissionLineUniverse
from daria.stats.fitting import lnpost
from daria.stats import MoveAssembler
from daria.line_lfs import Sobral13, Pang26, CoveloPaz25, Comparat16, \
    Khostovan15, Favole24

head_dir = 'fit_multiline_lfs'

### compile LFs into single dataset

dust = True

s13 = Sobral13(dust=dust)
data_Ha_lfs = s13.get_data()

cp25 = CoveloPaz25(dust=dust)
data_highz_Ha_lfs = cp25.get_data()

p26 = Pang26(dust=dust)
data_pang = p26.get_data()

c16 = Comparat16()
data_c16_hb = c16.get_data('Hb')
data_c16_o3 = c16.get_data('OIII_5007')
data_c16_o2 = c16.get_data('OII')

k15 = Khostovan15()
data_k15_hbo3 = k15.get_data('HbOIII')

f24 = Favole24(dust=True)
data_f24_ha = f24.get_data('Ha')
data_f24_hb = f24.get_data('Hb')
data_f24_o2 = f24.get_data('OII')
data_f24_o3 = f24.get_data('OIII_5007')
data_f24_n2 = f24.get_data('NII')

lf_label = 's13_cp25_p26_c16_k15_f24_all'

for z in list(data_highz_Ha_lfs.keys()):
    data_Ha_lfs[z] = data_highz_Ha_lfs[z]

data_Ha_lfs[0.1] = data_f24_ha[0.1]
data_c16_hb[0.1] = data_f24_hb[0.1]
data_c16_o2[0.1] = data_f24_o2[0.1]
data_c16_o3[0.1] = data_f24_o3[0.1]

data = {
    'fit_lf_lines': ['Ha','HaNII','NII','Hb','HbOIII','OIII_5007','OII'],
    'line_lfs': {'Ha': data_Ha_lfs,
                 'HaNII': data_pang,
                 'NII': data_f24_n2,
                 'Hb': data_c16_hb,
                 'HbOIII': data_k15_hbo3,
                 'OIII_5007': data_c16_o3,
                 'OII': data_c16_o2
                 }
    }

### define model

metal_line_model = 'indep_pow'
include_lines = ['Ha','Hb','NII','OIII_5007','OIII_4959','OII']

static_mmin_dict = {
    'mmin_z': 0,
    'rturn': np.inf
    }

mmin_str = ''
for i, (k,v) in enumerate(static_mmin_dict.items()):
    temp_str = f'{k}_{v}'
    mmin_str = os.path.join(mmin_str,temp_str)

if not dust:
    static_dust_dict = {
        'norm_Av_0': 0,
        'norm_Av_z': 0,
        'slope_Av': 0
        }
else:
    static_dust_dict = {}

dust_str = ''
for i, (k,v) in enumerate(static_dust_dict):
    temp_str = f'{k}_{v}'
    os.path.join(dust_str,temp_str)

static_properties = {
    'target_prop': {},
    'metal_line_model': metal_line_model,
    'include_lines': include_lines,
    'mask': None,
    'mlim': None
    }

static_properties.update(static_mmin_dict)
static_properties.update(static_dust_dict)

### initialize MCMC with some model; not sure i actually need to specify these

mcmc_init = {
    'mmin_0': 8,
    'norm_Av_0': 0.5,
    'norm_Av_z': -0.1,
    'slope_Av': 0.1,
    'N2_norm': -1,
    'N2_slope': 0.05,
    'O3_norm': 0,
    'O3_slope': 0.1,
    'norm_sfr_0': -2.5,
    'norm_sfr_a': 2.5,
    'norm_sfr_z': -0.1,
    'mbreak_sfr_0': 11,
    'mbreak_sfr_z': 0.1,
    'slope_lo_sfr': 0,
    'slope_hi_sfr': 1
    }

### define priors

priors = {
    'mmin_0': (6,10),
    'norm_Av_0': (0,2),
    'norm_Av_z': (-1,1),
    'slope_Av': (0,1),
    'N2_norm': (-2,-0.1),
    'N2_slope': (-0.5,0.5),
    'O3_norm': (-2,2),
    'O3_slope': (-0.5,0.5),
    'norm_sfr_0': (-6,0),
    'norm_sfr_a': (-5,5),
    'norm_sfr_z': (-3,3),
    'mbreak_sfr_0': (10,14),
    'mbreak_sfr_z': (-2,2),
    'slope_lo_sfr': (0,5),
    'slope_hi_sfr': (0,5)
    }

priors_str = ''
for i, (k,v) in enumerate(priors.items()):
    temp_str = f'{k}_{v[0]}_to_{v[1]}'
    priors_str = os.path.join(priors_str,temp_str)

### emcee-related settings

nwalkers = 256
nsteps = 100
nthreads = 1
clobber = False

if nthreads > 1:
    is_root_process = current_process().name == 'MainProcess'
else:
    is_root_process = True

emcee_moves = MoveAssembler()
moves = emcee_moves.assemble_moves()

moves_label = emcee_moves.label()
emcee_dir = os.path.join(moves_label,f'nwalkers{nwalkers}_nsteps{nsteps}')

dir_save = os.path.join(head_dir,lf_label,metal_line_model,mmin_str,\
                        dust_str,priors_str)
figdir = os.path.join(dir_save,emcee_dir)
os.makedirs(figdir,exist_ok=True)

fn_save = os.path.join(figdir,'data.pkl')
print(f'Will save results to:\n{fn_save}')

### initialize model and walker positions

params = list(priors.keys())

model_init_args = copy.deepcopy(mcmc_init)
model_init_args.update(static_properties)
model_init = ToyEmissionLineUniverse(**model_init_args)

if clobber or (not os.path.exists(fn_save)):
    new_run = True
    data_pre = None
    free_params = []
    for param in params:
        if param in ['slope_lo_sfr','slope_hi_sfr']:
            continue
        param_prior_lo, param_prior_hi = priors[param]
        init_param = np.random.uniform(low=param_prior_lo,high=param_prior_hi,\
                                       size=nwalkers)
        free_params.append(init_param)

    # eventually change in model to make +/-
    param_prior_lo, param_prior_hi = priors['slope_lo_sfr']
    init_param_lo = np.random.uniform(low=param_prior_lo,high=param_prior_hi,\
                                      size=nwalkers)
    init_param_hi = np.random.uniform(low=param_prior_lo,high=param_prior_hi,\
                                      size=nwalkers)
    free_params.append(np.min([init_param_lo,init_param_hi],axis=0))
    free_params.append(np.max([init_param_lo,init_param_hi],axis=0))

    pos = np.vstack(free_params).T

else:
    new_run = False
    fn_pos = fn_save
    with open(fn_pos,'rb') as f:
        data_pre = pickle.load(f)

    pos = data_pre['chain'][:,-1,:]
    print(f"Restarting from prev. output {fn_pos}.")
    print(f"Found {data_pre['fchain'].shape[0]} samples there.")
    print(f"Will augment with {nsteps} more steps per walker,")


### run MCMC

# is_root_process !!!
if nthreads > 1:
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers,pos.shape[1],lnpost,
                                        pool=pool,moves=moves,\
                                        args=[data,model_init,params,priors])
        if is_root_process:
            print(f"* Running with nthreads={nthreads}")
            print(f"* Starting {time.ctime()}")
            print(f"* Number of parameters: {len(params)}")

        t1 = time.time()
        results = sampler.run_mcmc(pos,nsteps,progress=True)
        t2 = time.time()

        if is_root_process:
            t_elapsed = (t2-t1)/60
            print(f"* Fit complete in {t_elapsed:.1f} minutes")
else:
    print(f"* Starting {time.ctime()}")
    print(f"* Will save results to {fn_save}")

    sampler = emcee.EnsembleSampler(nwalkers,pos.shape[1],lnpost,moves=moves,\
                                    args=[data,model_init,params,priors])

    t1 = time.time()
    results = sampler.run_mcmc(pos,nsteps,progress=True)
    t2 = time.time()
    t_elapsed = (t2-t1)/60
    print(f"Fit complete in {t_elapsed:.1f} minutes")

acceptance_fraction = sampler.acceptance_fraction
print(f"Mean acceptance fraction: {np.mean(acceptance_fraction):.3f}")
print(f"Median acceptance fraction: {np.median(acceptance_fraction):.3f}")

try:
    autocorr_time = np.mean(sampler.get_autocorr_time(tol=25))
except:
    autocorr_time = ' need more samples!!'
print(f"Mean autocorrelation time (steps): {autocorr_time}")

### save results

chain = sampler.chain
fchain = sampler.flatchain
lnprob = sampler.lnprobability
flnprob = sampler.flatlnprobability

# save file

if is_root_process:
    with open(fn_save,'wb') as f:
        if data_pre is None:
            out = {
                'chain': chain,
                'fchain': fchain,
                'lnprob': lnprob,
                'flnprob': flnprob,
                'data': data,
                'params': params,
                'acceptance_fraction': acceptance_fraction
                }
        else:
            out = {'chain': np.concatenate((data_pre['chain'],chain),axis=1),
                   'fchain': np.concatenate((data_pre['fchain'],fchain),\
                                            axis=1),
                   'lnprob': np.concatenate((data_pre['lnprob'],lnprob),\
                                            axis=1),
                   'flnprob': np.concatenate((data_pre['flnprob'],flnprob),\
                                             axis=1),
                   'data': data,
                   'params': params,
                   'acceptance_fraction': acceptance_fraction
                   }

        pickle.dump(out,f)

    print(f"Saved results to {fn_save}")
    print(f"{fchain.shape[0]} total samples.")

