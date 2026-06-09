import numpy as np

class Khostovan15(object):
    ''' Store and retrieve HiZELS Hb+[OIII] and [OII] luminosity functions.
    Data from Khostovan et al. 2015 '''
    def __init__(self):
        self.dust = True
        self.lines = ['HbOIII','OII']
        self.z = [[0.84,1.42,2.23,3.24],\
                  [1.47,2.25,3.34,4.69]]
        # data is [[[z1 of HbOIII], [z2 of HbOIII], ...],[[z1 of OII]...]]
        # this is to match Sobral13 format but may reformat both later
        self.log10L = \
            [ [ [41.1,41.3,41.5,41.7,41.9,42.1,42.3,42.5], \
                [41.195,42.25,42.55,42.85], \
                [42.6,42.75,42.9,43.05], \
                [42.65,42.8,42.95,43.1] ], \
              [ [41.65,41.8,41.95,42.1,42.25,42.4,42.55], \
                [42.45,42.65,42.85], \
                [43.05,43.15,43.3], \
                [42.86,43.01,43.16] ] ]
        ''' phi is corrected for completeness + filter shape. '''
        self.phi = \
            [ [ [-1.82,-2.04,-2.35,-2.61,-2.94,-3.17,-3.52,-4.12], \
                [-2.49,-3.14,-3.89,-4.64], \
                [-3.08,-3.14,-3.65,-4.26], \
                [-3.17,-3.26,-3.55,-4.17] ], \
              [ [-2.08,-2.28,-2.46,-2.69,-3.05,-3.55,-4.23], \
                [-2.77,-3.15,-4.46], \
                [-3.86,-3.92,-4.87], \
                [-3.66,-3.93,-4.11] ] ]
        self.phi_err = \
            [ [ [0.02,0.03,0.04,0.06,0.08,0.13,0.2,0.39], \
                [0.03,0.07,0.19,0.48], \
                [0.06,0.07,0.13,0.29], \
                [0.07,0.09,0.13,0.27] ], \
              [ [0.02,0.03,0.04,0.06,0.1,0.15,0.28], \
                [0.05,0.08,0.35], \
                [0.17,0.24,0.48], \
                [0.09,0.13,0.16] ] ]
                  
    def __get_attr_array(self,attr_name,idx,line_idx):
        attr = np.array(getattr(self,attr_name)[line_idx][idx])
        return attr
    
    def dict_keys(self):
        return ['log10L','phi','phi_err']
    
    def get_data(self,line='HbOIII'):
        assert line in self.lines, f'Choose from {self.lines}'
        line_idx = 0 if line == 'HbOIII' else 1
        data = {}
        z_dict_keys = self.dict_keys()
        for i, z in enumerate(self.z[line_idx]):
            z_dict = {}
            for key in z_dict_keys:
                z_dict[key] = self.__get_attr_array(key,i,line_idx)
            data[z] = z_dict
        return data

    def label(self,line='HbOIII'):
        return f'khostovan15_{line}'
