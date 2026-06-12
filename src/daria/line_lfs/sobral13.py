import numpy as np

class Sobral13(object):
    ''' Store and retrieve HiZELS Ha luminosity functions. Data from Sobral
    et al. 2013. '''
    def __init__(self,dust=False):
        self.Av = 1
        self.h = 0.7
        self.planck_h = 0.6766
        self.dust = dust
        self.include_NII = False
        self.line = 'Ha'
        self.z = [0.4,0.84,1.47,2.23]
        ''' log_LHa is corrected for [NII] contamination and dust extinction.
        The dust correction assumes uniform A_Ha = 1 magnitude. [NII]
        contamination removal is detailed in section 3.1 of Sobral+13 and
        basically amounts to correcting all Ha fluxes by relating the
        [NII]/Ha ratio to the combined EW. The median correction (i.e., the
        median [NII]/([NII]+Ha)) is ~0.25. '''
        self.log10L = \
            [ [40.5,40.6,40.7,40.8,40.9,41,41.1,41.2,41.3,41.4,41.5,\
               41.6,41.7,41.8,41.9,42,42.2,42.5], \
              [41.7,41.85,42,42.15,42.3,42.45,42.6,42.75,42.9], \
              [42.1,42.2,42.3,42.4,42.5,42.6,42.7,42.8,42.9,43,43.1,\
               43.2,43.4], \
              [42,42.15,42.3,42.4,42.5,42.6,42.7,42.8,42.9,43,43.1,43.2,\
               43.3,43.4,43.6] ]
        ''' phi_obs does not account for incompleteness and the shape of the
        narrowband filter profile. '''
        self.phi_obs = \
            [ [-1.84,-1.78,-1.87,-2.01,-2.2,-2.21,-2.41,-2.39,-2.43,-2.55,\
               -2.55,-2.71,-2.94,-2.9,-3.04,-3.34,-3.45,-3.64], \
              [-2.12,-2.11,-2.43,-2.72,-3.38,-3.46,-3.61,-4.16,-4.46], \
              [-2.2,-2.37,-2.55,-2.67,-2.78,-2.83,-3.23,-3.5,-3.91,-4.17,\
               -4.39,-4.57,-4.57], \
              [-2.18,-2.34,-2.24,-2.36,-2.48,-2.6,-2.68,-2.89,-3.18,-3.41,\
               -3.68,-4.04,-4.41,-4.59,-4.41] ]
        self.phi_obs_err = \
            [ [0.04,0.04,0.04,0.05,0.06,0.06,0.08,0.08,0.08,0.1,0.1,0.12,\
               0.17,0.16,0.19,0.3,0.36,0.53], \
              [0.03,0.03,0.04,0.06,0.15,0.17,0.21,0.53,0.9], \
              [0.1,0.08,0.06,0.05,0.05,0.04,0.07,0.1,0.18,0.26,0.37,\
               0.53,0.53], \
              [0.19,0.16,0.07,0.05,0.04,0.04,0.04,0.05,0.07,0.09,0.12,\
               0.21,0.37,0.53,0.37] ]
        ''' phi is corrected for completeness + filter shape. '''
        self.phi = \
            [ [-1.66,-1.7,-1.81,-1.93,-1.96,-2.03,-2.12,-2.27,-2.29,-2.42,\
               -2.46,-2.57,-2.69,-2.73,-2.88,-3.03,-3.56,-3.71], \
              [-1.93,-2.02,-2.18,-2.43,-2.73,-3.01,-3.27,-3.79,-4.13], \
              [-2.13,-2.25,-2.34,-2.47,-2.62,-2.73,-2.91,-3.18,-3.55,-3.81,\
               -4.22,-4.55,-4.86], \
              [-1.93,-2.07,-2.19,-2.31,-2.41,-2.5,-2.59,-2.73,-2.88,-3.09, \
               -3.33,-3.67,-4.01,-4.22,-4.63] ]
        self.phi_err = \
            [ [0.04,0.04,0.04,0.05,0.07,0.07,0.09,0.08,0.09,0.1,0.11,0.13, \
               0.19,0.17,0.2,0.35,0.51,0.71], \
              [0.03,0.03,0.04,0.06,0.17,0.17,0.21,0.55,1.51], \
              [0.1,0.09,0.06,0.05,0.05,0.04,0.08,0.11,0.18,0.26,0.38, \
               0.55,0.55], \
              [0.19,0.16,0.07,0.05,0.05,0.04,0.05,0.06,0.14,0.17,0.22,0.31, \
               0.51,0.68,0.41]]

    def get_dust_transmission(self):
        return 10**(-self.Av/2.5)
    
    def __get_attr_array(self,attr_name,idx):
        attr = np.array(getattr(self,attr_name)[idx])
        if (attr_name == 'log10L') and self.dust:
            transmission = self.get_dust_transmission()
            attr += np.log10(transmission) # (negative)
        elif attr_name == 'phi':
            attr += (3*np.log10(self.planck_h/self.h))
        return attr
    
    def dict_keys(self):
        return ['log10L','phi','phi_err']
    
    def get_data(self,dust=False):
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
        return f'sobral13_Ha_{dust_str}'
