import numpy as np
from importlib import resources

class SPHEREx(object):
    def __init__(self,noise='default'):
        self.noise = self.__get_noise(noise)

    def __get_noise(self,noise):
        if noise == 'default':
            noise_path = resources.files('daria') / \
                'instruments/data/SPHEREx_1sigma_noise.txt'
            noise = np.loadtxt(noise_path,skiprows=1)
            return noise
        else:
            return noise
        
    
