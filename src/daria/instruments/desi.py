import numpy as np

class DESI(object):
    def __init__(self,sample='bgs',release='dr1_lss',**kwargs):
        self.sample = sample.lower()
        if self.sample == 'bgs':
            self.subsample = kwargs.pop('subsample','all')
        self.release = release.lower()
        self.h = 0.6766
        self.dndV = self.__get_dndV()
        self.bias = self.__get_bias()
        
    def __get_dndV(self):
        sample = self.sample
        release = self.release
        if sample == 'bgs':
            if release == 'dr1_lss':
                # eyeball Fig 1 of DESI 2024 III
                z_ok = np.array([0.1,0.4])
                dndV = np.full(len(z_ok),3e-4 * self.h**3)
                return lambda z: np.interp(z,z_ok,dndV,left=0,right=0)
            else:
                if release == 'edr':
                    # eyeball Fig 19 of Hahn+23 (BGS final targ selection)
                    # (emphasis on eyeball)
                    subsample = self.subsample
                    z_ok = np.array([0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.65])
                    if subsample == 'all':
                        dndV = np.array([4e-1,5e-2,3e-2,1.5e-2,4e-3,1e-3,\
                                         1.5e-4,7e-5]) * self.h**3
                    elif subsample == 'bright':
                        dndV = np.array([4e-1,4e-2,2e-2,6e-3,1.5e-3,2e-4,\
                                         1e-5,0]) * self.h**3
                    elif subsample == 'faint':
                        dndV = np.array([1.5e-2,1e-2,1e-2,8e-3,3e-3,7e-4,\
                                         1e-4,5e-5]) * self.h**3
                    return lambda z: np.interp(z,z_ok,dndV,\
                                               left=0,right=0)
        else:
            raise AssertionError('unsupported sample')

    def __get_bias(self):
        if self.sample == 'bgs':
            return lambda z: 2
        else:
            raise AssertionError('unsupported sample')

    def get_target_prop(self):
        target_prop = {'dndV': self.dndV, 'bias': self.bias}
        return target_prop
    
    def label(self):
        sample = self.sample
        release = self.release
        if sample == 'bgs' and release == 'edr':
            sample = f'{sample}_{self.subsample}'
        label = f'desi_{sample}_{release}'
        return label
