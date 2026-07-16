import numpy as np
import pandas as pd
from importlib import resources
            
class Comparat16(object):
    def __init__(self):
        self.dust = True
        self.h = 0.6766
        self.planck_h = 0.6766
        self.lines = ['Hb','OIII_5007','OII']
        self.line_heads = ['H1_4862','O3_5007','O2_3728']
        z = []
        log10L = []
        phi = []
        phi_err = []
        for i,line in enumerate(self.lines):
            file_name = f'{self.line_heads[i]}-data-summary-Planck15.txt'
            fn = resources.files('daria') / f'line_lfs/data/comparat16/{file_name}'
            data = pd.read_csv(fn,sep=' ')
            # avoid some data sets I would rather not use
            data = data[data['Ngalaxy'] > 0]
            z_line = data['z_mean'].values
            z_line = np.unique(z_line)
            z.append(z_line)
            log10L_line = []
            phi_line = []
            phi_err_line = []
            for z_l in z_line:
                data_z = data[data['z_mean'] == z_l]
                log10L_z = np.log10(data_z['L_mean'].values)
                phi_z = np.log10(data_z['phi_mean'].values)
                phi_err_lo_z = np.log10(data_z['phi_mean'].values) - \
                    np.log10(data_z['phi_min'].values)
                phi_err_hi_z = np.log10(data_z['phi_max'].values) - \
                    np.log10(data_z['phi_mean'].values)
                phi_err_z = np.mean([phi_err_lo_z,phi_err_hi_z],axis=0)

                log10L_line.append(log10L_z)
                phi_line.append(phi_z)
                phi_err_line.append(phi_err_z)

            log10L.append(log10L_line)
            phi.append(phi_line)
            phi_err.append(phi_err_line)

        self.z = z
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

    def __get_line_idx(self,line='Hb'):
        if line.lower() == 'hb':
            return 0
        elif line.lower() == 'oiii_5007':
            return 1
        else:
            assert line.lower() == 'oii'
            return 2
        
    def get_data(self,line='Hb'):
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

    def label(self,line='Hb'):
        return f'comparat16_{line}'
