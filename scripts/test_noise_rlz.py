from daria.models import NoiseRlz,HelgasonModel,Population,Universe
from daria.utils import EllBinner,LinearDataBinner
from daria.instruments import SPHEREx, DESI
import numpy as np
import os

seed = 450293
ellbins = EllBinner()
zbins = LinearDataBinner(0.1,0.5,0.05)
field = 'deep'

hm = HelgasonModel()
sample = DESI()
target_prop = sample.get_target_prop()
mask_mag = 18
mask_wl = 1

universe = Universe(target_prop=target_prop,continuum_model=hm,\
                    mask_mag=mask_mag,mask_wl=mask_wl)

noise = NoiseRlz(seed,universe,ellbins,zbins,instrument=field)
