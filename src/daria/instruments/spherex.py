import numpy as np
import astropy.units as u
from importlib import resources

class SPHEREx(object):
    def __init__(self,field='deep',noise='default'):
        self.noise = self.__get_noise(noise)
        self.field = field.lower()
        self.fsky_deep = 200 * (np.pi/180)**2 / (4*np.pi)
        self.fsky_all = 0.7
        self.fsky = self.__get_fsky()

        # spectral resolution and channel binning
        self.band_R = [41,41,41,35,110,130] # spectral resolution per array
        self.chs_per_array = 17
        self.chs_tot = 51
        wl_bins, nu_bins = self.__get_channel_bins()
        self.wl_bins = wl_bins.value
        self.nu_bins = nu_bins.value

        # beam info
        self.reso = 6.2 * u.arcsecond        
        self.beam = (self.reso**2).to(u.sr)

        self.Cl_noise = self.__get_Cl_noise()
        
    def __get_fsky(self):
        field = self.field
        if field == 'deep':
            return self.fsky_deep
        elif field == 'all':
            return self.fsky_all
        else:
            assert field in ['deep_n','deep_s'], 'Invalid survey field'
            return self.fsky_deep * 0.5

    def __get_omega_field(self,unit='sr'):
        if unit.lower() == 'sr':
            return self.fsky * 4 * np.pi
        else:
            assert unit.lower() == 'deg2', 'Invalid solid angle unit'
            omega = self.__get_omega_field()
            return omega * (180/np.pi)**2
    
    def __get_noise(self,noise):
        if noise == 'default':
            noise_path = resources.files('daria') / \
                'instruments/data/SPHEREx/SPHEREx_1sigma_noise.txt'
            noise = np.loadtxt(noise_path,skiprows=1)
            return noise
        else:
            return noise

    def __get_channel_bins(self):
        noise_data = self.noise
        wl_bins = noise_data[:,0] * u.um
        nu_bins = wl_bins.to(u.GHz,equivalencies=u.spectral())
        return wl_bins, nu_bins
        
    def __get_Cl_noise(self):
        noise_idx = 1 if self.field == 'all' else 2
        Fnu = self.noise[:,noise_idx] * u.uJy
        Inu = Fnu / self.beam # uJy/sr
        nuInu = Inu * self.nu_bins * u.GHz # GHz uJy / sr
        sigma_noise = nuInu.to(u.nW/u.m**2/u.sr)
        Cl_noise = sigma_noise**2 * self.beam # nW2 m-4 sr-1
        return Cl_noise.value

class SPHERExCat(object):
    def __init__(self,zerr_type=2):
        self.zerr_type = zerr_type
        self.fsky = 0.7
        self.h = 0.6766

        self.nz_path = self.__get_path('nz')
        self.bias_path = self.__get_path('bias')

        self.dndV = self.__get_quantity('nz')
        self.bias = self.__get_quantity('bias')

    def __get_path(self,quantity):
        path = resources.files('daria') / \
            f'instruments/data/SPHEREx/spherex_public_{quantity}.csv'
        return path
        
    def __get_quantity(self,quantity):
        path = getattr(self,f'{quantity}_path')
        table = np.loadtxt(path,delimiter=',',skiprows=1)
        z_ok = table[:,0]
        z_ok[0] = table[0,1]
        z_ok[-1] = table[-1,2]
        quantity_ok = table[:,self.zerr_type+2]
        if quantity == 'nz':
            quantity_ok *= self.h**3 # provided in (h/Mpc)^3
        return lambda z: np.interp(z,z_ok,quantity_ok,left=0,right=0)

    def get_target_prop(self):
        target_prop = {}
        keys = ['dndV','bias','fsky']
        for key in keys:
            target_prop[key] = getattr(self,key)
        return target_prop

    def label(self):
        return f'spherexcat_zerr{self.zerr_type}'
