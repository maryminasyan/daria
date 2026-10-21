import numpy as np
from importlib import resources
import astropy.units as u

class DESI(object):
    '''
    Relevant properties of DESI DR1 LSS catalogs, i.e., n(z), bias, and fsky.
    '''
    def __init__(self,subdir='dr1_lss_v1.5',sample='BGS_BRIGHT',**kwargs):
        '''
        subdir: Subdirectory from which to retrieve n(z) info. Options:
            - `'dr1_lss_v1.5'`: n(z) from v1.5 DR1 LSS catalog directory:
                https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.5/
            - `'2024_bao'`: n(z) from Fig. 1 of JCAP 04 (2025) 012
        sample: DESI galaxy sample of interest. Options depend on subdir.
            - if subdir='dr1_lss_v1.5', options are 'BGS_ANY','BGS_BRIGHT',
              'BGS_BRIGHT-21.5', 'LRG', 'ELG_LOPnotqso', and 'QSO'
            - if subdir='2024_bao', options are 'BGS', 'LRG', 'ELG', and 'QSO'

        If using 'dr1_lss_v1.5', also need to provide kwarg:
            field: The survey field, either 'NGC' or 'SGC' (north/south
                galactic cap). Defaults to 'NGC'
        '''
        self.sample = sample
        self.subdir = subdir.lower()
        if self.subdir == 'dr1_lss_v1.5':
            assert self.sample in ['BGS_ANY','BGS_BRIGHT','LRG','QSO',\
                                   'BGS_BRIGHT-21.5','ELG_LOPnotqso']
            self.field = kwargs.pop('field','NGC').upper()
        else:
            assert self.subdir == '2024_bao'
            assert self.sample in ['BGS','ELG','LRG','QSO']
        self.catalog_fn = self.__assemble_catalog_fn()
        self.path = resources.files('daria') / \
            f'instruments/data/DESI/{self.subdir}/{self.catalog_fn}'
        self.is_csv = self.catalog_fn.endswith('csv')
        
        self.h = 0.6766
        self.dndV = self.__get_dndV()
        self.bias = self.__get_bias()
        self.fsky = self.__get_fsky()
        
    def __assemble_catalog_fn(self):
        if self.subdir == 'dr1_lss_v1.5':
            return f'{self.sample}_{self.field}_nz.txt'
        else:
            assert self.subdir == '2024_bao', 'Invalid subdir'
            return f'DESI_DR1_{self.sample}.csv'

    def __get_nz_file(self):
        skiprows = 0 if self.is_csv else 3
        delimiter = ',' if self.is_csv else None
        return np.loadtxt(self.path,delimiter=delimiter,skiprows=skiprows)
        
    def __get_dndV(self):
        subdir = self.subdir
        nz = self.__get_nz_file()
        if subdir == '2024_bao':
            z_ok = nz[:,0]
            dndV = nz[:,1] * self.h**3 # provided in (h/Mpc)^3
        else:
            z_ok = nz[:,0]
            # avoid interpolation boundary condition issues at edges of z range
            z_ok[0] = nz[0,1]
            z_ok[-1] = nz[-1,2]
            dndV = nz[:,3] * self.h**3 # provided in (h/Mpc)^3
        return lambda z: np.interp(z,z_ok,dndV,left=0,right=0)

    def __get_bias(self):
        sample_full = self.sample.lower()
        is_sample = lambda sample: sample_full.startswith(sample)
        # all but QSO from table 1 of arXiv:2601.05967
        if is_sample('bgs'):
            bias = lambda z: 1.5
        elif is_sample('lrg'):
            bias = lambda z: 2
        elif is_sample('elg'):
            bias = lambda z: 1.2
        elif is_sample('qso'):
            # from arXiv:2306.06314
            bias = lambda z: 2.6
        return bias

    def __get_fsky(self):
        if self.subdir == 'dr1_lss_v1.5':
            with open(self.path) as f:
                next(f) # full area
                line = next(f) # effective area
            area = float(line.split('is ')[1].split('square')[0])
        else:
            # numbers from table 2 of JCAP 07 (2025) 017
            if self.sample == 'BGS':
                area = 7473
            elif self.sample == 'LRG':
                area = 5740
            elif self.sample == 'ELG':
                area = 5924
            elif self.sample == 'QSO':
                area = 7249
        area = (area * u.deg**2).to(u.sr).value
        return area / (4*np.pi)

    def get_target_prop(self):
        target_prop = {
            'dndV': self.dndV,
            'bias': self.bias,
            'fsky': self.fsky
            }
        return target_prop

    def label(self):
        field = f'_{self.field}' if self.subdir == 'dr1_lss_v1.5' else ''
        return f'desi_{self.subdir}_{self.sample}{self.field}'
