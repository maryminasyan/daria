import numpy as np

class CoveloPaz25:
    ''' Ha LFs at z ~ 4-6.5, per Table 1 of
    https://doi.org/10.1051/0004-6361/202452363
    '''
    def __init__(self):
        self.dust = True
        self.h = 0.7
        self.planck_h = 0.6766
        self.line = 'Ha'
        self.z = [4.45,5.3,6.15]
        self.log10L = \
            [ [42,42.25,42.5,42.75,43,43.25],\
              [42,42.25,42.5,42.75,43],\
              [42,42.25,42.5,42.75] ]
        self.phi = \
            [ np.log10([3.75e-3,2.59e-3,0.9e-3,0.48e-3,0.096e-3,0.047e-3]),\
              np.log10([3.04e-3,1.82e-3,0.97e-3,0.366e-3,0.142e-3]),\
              np.log10([1.18e-3,0.72e-3,0.38e-3,0.12e-3]) ]
        self.phi_exp_err_hi = \
            [ [0.6,0.44,0.21,0.15,0.077,0.062],\
              [0.48,0.32,0.2,0.118,0.079],\
              [0.23,0.17,0.15,0.075] ]
        self.phi_exp_err_lo = \
            [ [0.59,0.43,0.19,0.13,0.048,0.031],\
              [0.47,0.3,0.19,0.098,0.056],\
              [0.34,0.22,0.12,0.067] ]
        self.phi_err_lo = \
            [ self.phi[i] - np.log10(10**self.phi[i] - \
                                     1e-3*np.array(self.phi_exp_err_lo[i])) \
              for i in range(len(self.z)) ]
        self.phi_err_hi = \
            [ np.log10(10**self.phi[i] + \
                       1e-3*np.array(self.phi_exp_err_hi[i])) - \
              self.phi[i] for i in range(len(self.z)) ]
    
    def __get_attr_array(self,attr_name,idx):
        if attr_name == 'phi_err':
            err_lo = self.__get_attr_array('phi_err_lo',idx)
            err_hi = self.__get_attr_array('phi_err_hi',idx)
            attr = np.mean([err_lo,err_hi],axis=0)
        else:
            attr = np.array(getattr(self,attr_name)[idx])
            if attr_name == 'phi':
                attr += (3*np.log10(self.planck_h/self.h))
            
        return attr
    
    def dict_keys(self):
        return ['log10L','phi','phi_err']#_lo','phi_err_hi']
    
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
        return f'covelo-paz25_Ha'
