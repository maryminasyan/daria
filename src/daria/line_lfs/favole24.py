import numpy as np
import pandas as pd
from importlib import resources
            
class Favole24(object):
    def __init__(self,population='full',dust=True):
        '''
        Populations: full, sf_bpt, sf_ssfr, passive, composite, liner, seyfert
        '''
        self.population = population.lower()
        self.dust = dust
        dust_str = 'observed' if self.dust else 'intrinsic'
        file_name = f'{self.population}_{dust_str}_lfs.txt'
        fn = resources.files('daria') / f'line_lfs/data/favole24/{file_name}'
        
        self.h = 0.6777
        self.planck_h = 0.6766

        self.lines = ['Ha','Hb','OII','OIII_5007','NII']
        line_labels = ['Ha','Hb','O2','O3','N2']
        self.z = [[0.1] for line in self.lines]

        log10L = []
        phi = []
        phi_err = []

        data = pd.read_csv(fn,sep=';',skiprows=[1,2])
        for i,line_label in enumerate(line_labels):
            log10L_line = [data['logL'].values]
            
            phi_lin = data[f'LF{line_label}'].values

            phi_elin = data[f'e_LF{line_label}'].values
            phi_err_low = np.log10(phi_lin + phi_elin) - np.log10(phi_lin)
            phi_err_hi = np.log10(phi_lin) - np.log10(phi_lin - phi_elin)

            phi_line = [np.log10(phi_lin)]
            phi_err_line = [np.mean([phi_err_low,phi_err_hi],axis=0)]

            log10L.append(log10L_line)
            phi.append(phi_line)
            phi_err.append(phi_err_line)

        self.log10L = log10L
        self.phi = phi
        self.phi_err = phi_err
                  
    def __get_attr_array(self,attr_name,idx,line_idx):
        attr = np.array(getattr(self,attr_name)[line_idx][idx])
        if attr_name == 'log10L':
            attr += (2*np.log10(self.h/self.planck_h))
        elif attr_name == 'phi':
            attr += (3*np.log10(self.planck_h/self.h))
        return attr
    
    def dict_keys(self):
        return ['log10L','phi','phi_err']

    def __get_line_idx(self,line='Ha'):
        if line.lower() == 'ha':
            return 0
        elif line.lower() == 'hb':
            return 1
        elif line.lower() == 'oii':
            return 2
        elif line.lower() == 'oiii_5007':
            return 3
        elif line.lower() == 'nii':
            return 4
        
    def get_data(self,line='Ha'):
        assert line in self.lines, f'Choose from {self.lines}'
        line_idx = self.__get_line_idx(line)
        data = {}
        z_dict_keys = self.dict_keys()
        for i, z in enumerate(self.z[line_idx]):
            z_dict = {}
            for key in z_dict_keys:
                z_dict[key] = self.__get_attr_array(key,i,line_idx)
            data[z] = z_dict
        return data

    def label(self,line='Ha'):
        dust_str = '_obs' if self.dust else '_intrr'
        return f'favole24_{line}{dust_str}'
