import numpy as np
from .lf_tools import correct_little_h

class Meyer24(object):
    ''' Store and retrieve [OIII] LFs from Meyer et al 2024 '''
    def __init__(self,field='combined'):
        self.field = field.lower()
        # (GOODS-N/S/combined)
        assert self.field in ['north','south','combined'], \
            'Invalid survey field'
        self.h = 0.7
        self.planck_h = 0.6766
        self.dust = True
        self.line = 'OIII_5007'
        self.z = [7.1,7.9]
        ''' Some of these points are upper limits and phi_err will be
        treated incorrectly. For now just stick to combined fields to
        circumvent this issue. '''
        # check completeness !
        if self.field == 'north':
            self.log10L = [ [41.6,41.8,42,42.2,42.4,42.6,42.8,43], \
                            [41.8,42,42.2,42.4,42.6,42.8,43,43.2] ]
            self.phi = [ [-1.98,-2.51,-2.25,-2.77,-3.03,-3.23,-3.93,-3.64],\
                         [-3.13,-2.45,-3.03,-3.19,-3.89,-4.19,-4.15,-4.15] ]
            self.phi_err_lo = [ [1.26,0.8,0.24,0.08,0.09,0.13,0.36,0.24],\
                                [0.46,10,0.15,0.11,0.22,0.24,0.37,0.38] ]
            self.phi_err_hi = [ [0.29,0.26,0.15,0.07,0.1,0.13,0.33,0.23],\
                                [0.22,0.33,0.11,0.1,0.23,0.33,0.34,0.34] ]
        elif self.field == 'south':
            self.log10L = [ [41.6,41.8,42,42.2,42.4,42.6,42.8,43,43.2],\
                            [41.8,42,42.2,42.4,42.6,42.8,43] ]
            self.phi = [ [-2.43,-2.73,-2.95,-3.07,-3.62,-3.87,-4.24,-4.25,\
                          -4.25],\
                         [-2.93,-2.33,-2.54,-3.1,-3.81,-3.8,-4.44] ]
            self.phi_err_lo = [ [10,0.61,0.1,0.18,0.19,0.29,0.59,0.61,0.62],\
                                [0.67,0.64,0.2,0.11,0.22,0.21,0.55] ]
            self.phi_err_hi = [[0.46,0.24,0.09,0.14,0.21,0.3,0.49,0.49,0.49],\
                               [0.25,0.25,0.13,0.1,0.22,0.21,0.48] ]
        elif self.field == 'combined':
            self.log10L = [ [41.6,41.8,42,42.2,42.4,42.6,42.8,43,43.2], \
                            [41.8,42,42.2,42.4,42.6,42.8,43,43.2]]
            self.phi = [ [-2.15,-2.61,-2.47,-2.89,-3.23,-3.44,-4.06,-3.85,\
                          -4.55],\
                         [-3.02,-2.39,-2.72,-3.14,-3.85,-3.95,-4.27,-4.45] ]
            self.phi_err_lo = [ [0.84,0.39,0.19,0.07,0.08,0.11,0.28,0.21,\
                                 0.62],\
                                [0.34,0.47,0.14,0.08,0.15,0.16,0.28,0.38] ]
            self.phi_err_hi = [ [0.27,0.2,0.13,0.07,0.08,0.12,0.27,0.21,0.49],\
                                [0.19,0.22,0.11,0.07,0.15,0.17,0.27,0.34] ]
            
    def __get_attr_array(self,attr_name,idx):
        if attr_name == 'phi_err':
            err_lo = self.__get_attr_array('phi_err_lo',idx)
            err_hi = self.__get_attr_array('phi_err_hi',idx)
            attr = np.mean([err_lo,err_hi],axis=0)
        else:
            attr = np.array(getattr(self,attr_name)[idx])
        attr = correct_little_h(attr,attr_name,h_in=self.h)
        return attr
    
    def dict_keys(self):
        return ['log10L','phi','phi_err']
    
    def get_data(self):
        data = {}
        z_dict_keys = self.dict_keys()
        for i, z in enumerate(self.z):
            z_dict = {}
            for key in z_dict_keys:
                z_dict[key] = self.__get_attr_array(key,i)
            data[z] = z_dict
        return data
        
    def label(self):
        return 'meyer24'
