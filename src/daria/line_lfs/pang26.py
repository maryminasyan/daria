import numpy as np

class Pang26(object):
    ''' Store and retrieve GLASS-JWST Ha+[NII] LFs. Data is from Pang et al.
    2026. '''
    def __init__(self,dust=False):
        self.Av = 1
        self.h = 0.7
        self.planck_h = 0.6766
        self.dust = dust
        self.include_NII = True # redundant arg?
        self.line = 'HaNII'
        self.z = [1.3,2.0]
        self.log10L = \
            [ [40.75,41.25,41.75,42.25,42.75], \
              [41.1,41.5,41.9,42.3] ]
        self.phi = \
            [ [-1.57,-2.0,-2.19,-2.79,-3.27], \
              [-1.92,-2.06,-2.44,-2.9] ]
        # split normal likelihood?
        self.phi_err_lo = \
            [ [0.14,0.14,0.17,0.38,1.06], \
              [0.24,0.15,0.28,0.56] ]
        self.phi_err_hi = \
            [ [0.13,0.12,0.15,0.3,0.52], \
              [0.21,0.13,0.23,0.37] ]

    def get_dust_transmission(self):
        return 10**(-self.Av/2.5)
    
    def __get_attr_array(self,attr_name,idx):
        if attr_name == 'phi_err':
            err_lo = self.__get_attr_array('phi_err_lo',idx)
            err_hi = self.__get_attr_array('phi_err_hi',idx)
            attr = np.mean([err_lo,err_hi],axis=0)
        else:
            attr = np.array(getattr(self,attr_name)[idx])
            if (attr_name == 'log10L') and self.dust:
                transmission = self.get_dust_transmission()
                attr += np.log10(transmission) # (negative)
            elif attr_name == 'phi':
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
        dust_str = '_dust' if self.dust else ''
        return f'pang26_HaNII_{dust_str}'
