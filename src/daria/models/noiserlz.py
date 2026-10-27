import numpy as np
import os
from ..utils import LinearDataBinner
from ..bookkeeping import Config
from ..instruments import instrument_tools as it
from .model_tools import get_conti_mask_label

class NoiseRlz(object):
    '''
    Generate a realistic, if optimistic, noise realization
    '''
    def __init__(self,seed,universe,ellbinner,zbinner,instrument='deep',\
                 wlbinner=None):
        '''
        seed [int] RNG seed
        universe [Universe object]
        ellbinner [EllBinner object]
        zbinner [LinearDataBinner object]
        instrument [str or SPHEREx object]; can provide field
            ('deep','deep_n','deep_s',or 'all') as shortcut and will
            auto-generate object
        wlbinner [LinearDataBinner object, optional]: provide if not using
            default SPHEREx public channel definitions
        '''
        self.seed = seed
        self.universe = universe
        self.instrument = it.get_instrument(instrument)
        self.ell_centers = ellbinner.ell_bins
        self.nell = ellbinner.nell
        self.wl_edges = self.__get_wl_edges(wlbinner)
        self.z_edges = zbinner.edges(dim=2)
        self.array_shape = (len(self.wl_edges),len(self.z_edges),\
                            len(self.ell_centers))
        self.fsky = self.__get_fsky()

        self.wl_label = self.__get_wl_label(wlbinner)
        self.z_label = zbinner.label('z')
        self.ell_label = ellbinner.label()
        self.fn = self.__get_fn()
        self.noise = self.__get_noise_dell()

    def __get_wl_edges(self,wlbinner):
        if wlbinner is None:
            return self.instrument.get_wl_edges()
        else:
            return wlbinner.edges(dim=2)

    def __get_wl_label(self,wlbinner):
        if wlbinner is None:
            return ''
        else:
            return wlbinner.label('wl')
        
    def __get_fsky(self):
        spherex_fsky = self.instrument.fsky
        catalog_fsky = self.universe.target_prop['fsky']
        return np.min([spherex_fsky,catalog_fsky])
    
    def __get_unity_array(self):
        return np.ones(self.array_shape)

    def __get_cell_x_lines(self):
        return self.universe.get_xcorr_tot(self.ell_centers,self.wl_edges,\
                                           self.z_edges,sum_lines=True,\
                                           output='cell')

    def __get_cell_i_inst(self):
        cell_i_inst_mtx = self.instrument.Cl_noise
        unity = self.__get_unity_array()
        return cell_i_inst_mtx[:,None,None] * unity

    def __get_cell_i_lines(self):
        cell_i_lines_mtx = self.universe.get_ps_tot(self.ell_centers,\
                                                    self.wl_edges,\
                                                    sum_lines=True,\
                                                    output='cell')
        unity = self.__get_unity_array()
        return cell_i_lines_mtx.T.diagonal()[:,None,:] * unity

    def __get_cell_i_conti(self):
        zbins_conti = LinearDataBinner(0.05,10,0.1)
        z_edges_2d_conti = zbins_conti.edges(dim=2)
        fn = self.__get_conti_fn(zbins_conti)
        if os.path.exists(fn):
            print(f'Retrieving continuum auto from:\n{fn}')
            cell_i_conti_mtx = np.load(fn)
        else:
            print('Generating continuum auto...')
            head, tail = os.path.split(fn)
            cell_i_conti_mtx = self.universe.\
                get_conti_ps_tot(self.ell_centers,z_edges_2d_conti,\
                                 self.wl_edges,output='cell')
            print('...finished generating continuum auto...')
            os.makedirs(head,exist_ok=True)
            np.save(fn,cell_i_conti_mtx)
            print(f'...successfully saved to:\n{fn}')
        unity = self.__get_unity_array()
        return cell_i_conti_mtx[:,None,:] * unity

    def __get_cell_x_conti(self):
        cell_x_conti = self.universe.get_conti_xcorr_tot(self.ell_centers,\
                                                         self.wl_edges,\
                                                         self.z_edges,\
                                                         output='cell')
        return cell_x_conti

    def __get_cell_g(self):
        cell_g_mtx = self.universe.get_gal_ps_tot(self.ell_centers,\
                                                  self.z_edges,\
                                                  output='cell')
        unity = self.__get_unity_array()
        return cell_g_mtx[None,:,:] * unity

    def __cell_to_dell(self):
        ell_centers = self.ell_centers
        unity = self.__get_unity_array()
        cell_to_dell_mtx = ell_centers * (ell_centers + 1) / (2 * np.pi)
        return cell_to_dell_mtx[None,None,:] * unity
        
    def get_var_cell(self):
        cell_i_conti = self.__get_cell_i_conti()
        cell_i_lines = self.__get_cell_i_lines()
        cell_i_inst = self.__get_cell_i_inst()
        cell_i = cell_i_conti + cell_i_lines + cell_i_inst

        cell_x_conti = self.__get_cell_x_conti()
        cell_x_lines = self.__get_cell_x_lines()
        cell_x = cell_x_conti + cell_x_lines

        cell_g = self.__get_cell_g()
        nell = self.nell[None,None,:] * self.__get_unity_array()

        var_cell = (1/(nell*self.fsky)) * (cell_x**2 + cell_i * cell_g)
        return var_cell

    def get_std_dell(self):
        cell_to_dell = self.__cell_to_dell()
        var_cell = self.get_var_cell()
        std_dell = cell_to_dell * np.sqrt(var_cell)
        return std_dell

    def get_snr(self):
        signal = self.__get_cell_x_lines()
        noise = self.get_var_cell()
        return signal/np.sqrt(noise)
        
    def __get_noise_dell(self):
        fn = self.fn
        if os.path.exists(fn):
            print(f'Retrieving noise rlz from:\n{fn}')
            noise = np.load(fn)
        else:
            print(f'Generating noise rlz with seed {self.seed}...')
            rng = np.random.default_rng(seed=self.seed)
            std_dell = self.get_std_dell()
            noise = rng.normal(scale=std_dell)
            print('...successfully generated noise rlz...')
            head, tail = os.path.split(fn)
            os.makedirs(head,exist_ok=True)
            np.save(fn,noise)
            print(f'...saved noise rlz to:\n{fn}')
        return noise

    def __get_fn(self):
        tail = f'seed{self.seed}.npy'
        fsky_label = f'fsky{self.fsky:.4f}'
        c = Config()
        fn = os.path.join(c.noise_dir,self.universe.label(),\
                          self.instrument.label(),\
                          fsky_label,self.wl_label,self.z_label,\
                          self.ell_label,tail)
        return fn

    def __get_conti_fn(self,zbins_conti):
        tail = 'continuum.npy'
        c = Config()
        conti_label, mask_label = get_conti_mask_label(self.universe)
        fn = os.path.join(c.conti_dir,conti_label,mask_label,self.wl_label,\
                          zbins_conti.label('z'),self.ell_label,tail)
        return fn
