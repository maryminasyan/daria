import numpy as np

class Matthee23(object):
    ''' Store and retrieve [OIII] LFs from Matthee et al 2023 '''
    def __init__(self):
        self.h = 0.674
        self.planck_h = 0.6766
        self.dust = True
        self.line = 'OIII_5007'
        # note their discussion about masking z range near bright quasar
        self.z = [6]
        self.log10L = [42,42.2,42.4,42.6,42.8,43,43.2,43.4]
        # check completeness !
        self.phi = [-2.67,-2.63,-2.83,-3.16,-3.34,-3.34,-3.66,-3.9]
        self.phi_err_lo = [0.2,0.16,0.12,0.17,0.19,0.19,0.30,0.53]
        self.phi_err_hi = [0.18,0.14,0.1,0.13,0.13,0.13,0.18,0.23]
    
    def __get_attr_array(self,attr_name):
        if attr_name == 'phi_err':
            err_lo = self.__get_attr_array('phi_err_lo')
            err_hi = self.__get_attr_array('phi_err_hi')
            attr = np.mean([err_lo,err_hi],axis=0)
        else:
            attr = np.array(getattr(self,attr_name))
            if (attr_name == 'log10L'):
                attr += (2*np.log10(self.h/self.planck_h))
            elif attr_name == 'phi':
                attr += (3*np.log10(self.planck_h/self.h))
        return attr
    
    def dict_keys(self):
        return ['log10L','phi','phi_err']
    
    def get_data(self):
        data = {}
        z_dict_keys = self.dict_keys()
        for i, z in enumerate(self.z):
            z_dict = {}
            for key in z_dict_keys:
                z_dict[key] = self.__get_attr_array(key)
            data[z] = z_dict
        return data
        
    def label(self):
        return 'matthee23'
