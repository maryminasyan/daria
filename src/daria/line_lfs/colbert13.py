import numpy as np
from .lf_tools import correct_little_h

class Colbert13(object):
    ''' Colbert et al 2013 Ha and OIII LFs from WISP '''
    def __init__(self):
        self.dust = True # provide observed LFs
        self.h = 0.7
        self.planck_h = 0.6766
        self.lines = ['Ha','OIII']
        self.z = [[0.6,1.2],\
                  [1.1,1.9]]
        # data is [[[z1 of Ha], [z2 of Ha]], [[z1 of OIII],[z2 of OIII]]]
        # this is to match Sobral13 format but may reformat both later
        self.log10L = \
            [ [ [40.55,40.85,41.15,41.45,41.75,42.05,42.35],\
                [41.15,41.45,41.75,42.05,42.35,42.75,42.95,43.25]],\
              [ [41.45,41.75,42.05,42.35,42.65,42.95],\
                [42.05,42.35,42.65,42.95,43.25,43.55]] ]
        ''' phi is corrected for completeness '''
        self.phi = \
            [ [ np.log10([0.0145,0.0109,0.008,0.0052,0.0025,0.0007,8.9e-5]),\
                np.log10([0.0148,0.007,0.0056,0.0025,0.00087,0.00014,5.8e-5,\
                          1.6e-5])],\
              [ np.log10([0.0016,0.002,0.0012,0.00055,0.00013,1.8e-5]),\
                np.log10([0.00059,0.00077,0.00036,0.00016,2.2e-5,2.2e-5])] ]
        given_err = \
            [ [ np.array([0.0024,0.0016,0.001,0.0007,0.0004,0.0002,6.5e-5]),\
                np.array([0.0036,0.001,0.0008,0.0003,0.00016,5.5e-5,3.5e-5,\
                          1.7e-5]) ],\
              [ np.array([0.00034,0.00032,0.0002,0.00012,5.3e-5,1.8e-5]),\
                np.array([0.0002,0.00021,0.0001,6.5e-5,1.6e-5,1.6e-5]) ] ]
        self.phi_err_hi = \
            [ [np.log10(10**(self.phi[j][i]) + given_err[j][i]) - \
               np.log10(10**(self.phi[j][i])) for i in \
               range(len(self.phi[j]))] for j in range(2) ]
        self.phi_err_lo = \
           [ [np.log10(10**(self.phi[j][i])) - \
              np.log10(10**(self.phi[j][i]) - given_err[j][i]) for i in \
              range(len(self.phi[j]))] for j in range(2) ]
        # phi err > phi --> lower bound becomes nan for one of the pts ...
        self.phi_err_lo[0][1][-1] = np.inf
                  
    def __get_attr_array(self,attr_name,idx,line_idx):
        if attr_name == 'phi_err':
            err_lo = self.__get_attr_array('phi_err_lo',idx,line_idx)
            err_hi = self.__get_attr_array('phi_err_hi',idx,line_idx)
            attr = np.mean([err_lo,err_hi],axis=0)
        else:
            attr = np.array(getattr(self,attr_name)[line_idx][idx])
        attr = correct_little_h(attr,attr_name,h_in=self.h)
        return attr
    
    def dict_keys(self):
        return ['log10L','phi','phi_err']
    
    def get_data(self,line='Ha'):
        assert line in self.lines, f'Choose from {self.lines}'
        line_idx = 0 if line == 'Ha' else 1
        data = {}
        z_dict_keys = self.dict_keys()
        for i, z in enumerate(self.z[line_idx]):
            z_dict = {}
            for key in z_dict_keys:
                z_dict[key] = self.__get_attr_array(key,i,line_idx)
            data[z] = z_dict
        return data

    def label(self,line='Ha'):
        return f'colbert13_{line}'
