import numpy as np
from functools import cached_property
from hmf import MassFunction
from ..utils.line_info import line_names
from .toygalaxypopulation import ToyGalaxyPopulation
from . import model_tools as mt

class ToyEmissionLineUniverse(object):
    """
    This class streamlines modeling of galaxy/EBL cross-correlations over the
    full wavelength and redshift range. It works in terms of parameters that
    describe the redshift evolution of the SFR(Mh) relationship and dust
    correction, whereas the ToyGalaxyPopulation does not explicitly deal
    in redshift evolution.
    """
    def __init__(self,target_prop=None,metal_line_model='l_sfr',mlim=11,\
                 mask=12,include_lines='all',mmin_0=8,mmin_z=0,\
                 rturn=1,norm_sfr_0=-4,norm_sfr_a=2.5,norm_sfr_z=0,\
                 mbreak_sfr=12,slope_lo_sfr=1.5,slope_hi_sfr=0.5,\
                 norm_Av_0=1,norm_Av_z=0.2,slope_Av=0.1,**kwargs):
        """
        Note - mlim, mask, mmin_0, and norm_sfr_0 are all LOG quantities.
        
        metal_line_model : str, bool
            How to model metal lines (mainly [NII] and [OIII], but [OII] is
            affected by association). Depending on the model you choose, you
            will need to provide different kwargs. Options are:
            - 'l_sfr' or True: Apply line luminosity-SFR scaling. PRovide no
            additional kwargs.
            - 'indep_pow': [NII]/Ha and [OIII]/Hb ratios have independent
            power law halo mass prescriptions.
            - 'o3_dep_n2': [NII]/Ha is a power law of halo mass. [OIII]/Hb
            scales with [NII]/Ha according to some empirical formula with
            tunable parameters.
            - False: No metal lines. Provide no additional kwargs.

        kwargs: For some metal line models, you need to provide more
        parameters.

            If `metal_line_model='indep_pow'`, provide:
                N2_norm : float
                    NII/Ha ratio in a 10^10 Msun halo
                N2_slope : float
                    Slope of N2-Mh relation
                O3_norm : float
                    OIII/Hb ratio in a 10^10 Msun halo
                O3_slope : float
                    Slope of O3-Mh relation

            If `metal_line_model='o3-dep-n2'`, O3 scales with N2 as
            O3 = O3_const + (O3_num / (N2 + O3_denom)). Therefore, provide:
                N2_norm : float
                    NII/Ha ratio in a 10^10 Msun halo
                N2_slope : float
                    Slope of N2-Mh relation
                O3_const : float
                O3_num : float
                O3_denom : float
        """
        self.target_prop = target_prop
        self.metal_line_model = mt.get_metal_line_model(metal_line_model)
        self.mlim = mt.get_mlim(mlim)
        self.mask = mt.get_mask(mask)
        if include_lines == 'all':
            self.include_lines = line_names
        else:
            self.include_lines = include_lines

        # Model parameters
        self.mmin_0 = mmin_0
        self.mmin_z = mmin_z
        self.rturn = rturn
        self.norm_sfr_0 = norm_sfr_0
        self.norm_sfr_a = norm_sfr_a
        self.norm_sfr_z = norm_sfr_z
        self.mbreak_sfr = mbreak_sfr
        self.slope_lo_sfr = slope_lo_sfr
        self.slope_hi_sfr = slope_hi_sfr
        self.norm_Av_0 = norm_Av_0
        self.norm_Av_z = norm_Av_z
        self.slope_Av = slope_Av

        # Metal line parameters depend on the chosen model
        mt.set_attrs(self,mt.get_default_dict(self.metal_line_model),**kwargs)

        ''' Create a dictionary with the initial parameter inputs. This
        is useful if we update any parameters later on (to avoid the cost of
        re-initializing the entire class). Mind that this stores the user-
        provided inputs, NOT the default values. This also does not store
        input args like `target_prop` or `metal_line_model`, which will not be
        updated. '''
        init_args = {key:value for key, value in locals().items() \
                     if key not in ['self','target_prop','metal_line_model',\
                                    'mlim','mask','include_lines','kwargs']}

        # Add individual kwargs entries; assume user defined them correctly...
        init_args.update(**kwargs)
        self.init_args = init_args
        self.init_kwargs = kwargs # Also need kwargs saved separately

        # Initialize `pop` with these args so it has corresponding `init_args`
        self.pop = ToyGalaxyPopulation(target_prop=self.target_prop,\
                                       metal_line_model=self.metal_line_model,\
                                       mlim=self.mlim,mask=self.mask,\
                                       mmin=self.mmin_0,rturn=self.rturn,\
                                       norm_sfr=self.norm_sfr_0,\
                                       slope_lo_sfr=self.slope_lo_sfr,\
                                       slope_hi_sfr=self.slope_hi_sfr,\
                                       norm_Av=self.norm_Av_0,\
                                       slope_Av=self.slope_Av,**kwargs)

    def update(self,**kwargs):
        """
        Update parameters with kwargs.

        It may become expensive to repeatedly initialize this class during
        likelihood evaluation. You can avoid this cost by initializing once
        per thread and updating parameters as needed. Make sure that parallel
        tasks are not pointing to the same memory space to avoid conflicting
        parameter updates. `multiprocess.pool` should automatically avoid
        this, but there could still be room for similar issues.
        """
        assert all(k not in self.zdep_attrs for k in kwargs.keys()), \
            'It looks like you provided 1+ args for ToyGalaxyPopulation.'\
            '\nPlease only provide args for ToyEmissionLineUniverse!'

        mt.set_attrs(self,self.init_args,overwrite=True,**kwargs)

        # Also need to update `ToyGalaxyPopulation` for consistency.

        # 1. Get kwargs that have exact matches for `ToyGalaxyPopulation` args
        pop_kwargs = {key:value for key, value in kwargs.items() \
                      if key in self.pop.init_args.keys()}

        # 2. "Massage" other relevant kwargs to have correct attr names (e.g.,
        # things like `mmin_0` will be assigned to `mmin` in `pop`)
        suffix = '_0'
        for key, value in kwargs.items():
            if key.endswith(suffix):
                pop_kwargs[key.removesuffix(suffix)] = value

        # 3. Update
        self.pop.update(**pop_kwargs)

    @cached_property
    def zdep_attrs(self):
        """
        Any parameters in `ToyGalaxyPopulation` that have some redshift
        scaling.
        """
        attrs = self.init_args.keys()
        zdep = []
        suffix = '_z'
        for attr in attrs:
            if attr.endswith(suffix):
                zdep.append(attr.removesuffix(suffix))
        return zdep

    def get_a(self,z):
        return 1/(1+z)

    def __get_a_pow(self,z,c_0,c_z):
        a = self.get_a(z)
        return c_0 + a**-c_z

    def get_mmin(self,z):
        ''' log10(mmin) '''
        return np.log10(self.__get_a_pow(z,10**self.mmin_0,self.mmin_z))

    def get_norm_Av(self,z):
        ''' (not a log) '''
        norm_Av = self.__get_a_pow(z,self.norm_Av_0,self.norm_Av_z)
        return max(norm_Av,0)

    def get_norm_sfr(self,z):
        ''' Implicitly log10(norm_sfr) '''
        a = self.get_a(z)
        norm_sfr = self.norm_sfr_0 + \
            self.norm_sfr_a * (1. - a) + self.norm_sfr_z * z
        return norm_sfr

    def update_pop_z(self,z):
        """
        Perform redshift scaling on any relevant parameters in
        `ToyGalaxyPopulation`. Currently, only `mmin`, `norm_sfr`, and
        `norm_Av` are redshift-dependent. This does **NOT** update `hmf`, nor
        does it change any parameters of `ToyEmissionLineUniverse`.

        To make some other parameter `foo` redshift-dependent, you must have a
        variable `foo_z` defined upon initializing `ToyEmissionLineUniverse`,
        as well as a public method `get_foo(z)` that will perform the redshift
        scaling using `foo_z` and any other `foo`-related args.
        """
        for attr in self.zdep_attrs:
            setattr(self.pop,attr,getattr(self,f'get_{attr}')(z))

    def get_sfr(self,z):
        self.update_pop_z(z)
        return self.pop.get_sfr(self.pop.m)

    def get_sfrd(self,z):
        if type(z) != np.ndarray:
            z = np.array([z])

        sfrd = np.zeros_like(z)
        for i, _z_ in enumerate(z):
            self.update_pop_z(_z_)
            sfrd[i] = self.pop.get_sfrd(_z_)

        return sfrd

    def get_lf_line(self,z,line='Ha'):
        """
        Compute the line luminosity function at given `z`.
        """
        self.update_pop_z(z)
        return self.pop.get_lf_line(z,line=line)

    def __convert_output(self,ps,ell,output='cell'):
        if output.lower() == 'dell':
            conversion = ell[None,None,:] * (ell[None,None,:] + 1) / 2 / np.pi
        else:
            assert output.lower() == 'cell', 'Unsupported output'
            conversion = 1
        return ps * conversion

    def get_conti_ps_tot(self,ell,zbins,channels,continuum_model,\
                         output='cell',**kwargs):
        conti_ps_mtx = np.zeros((channels.shape[0],ell.size))
        zbins_c = np.mean(zbins,axis=1)
        channels_c = np.mean(channels,axis=1)
        for j, wl in enumerate(channels_c):
            for i, zbin in enumerate(zbins):
                z = zbins_c[i]
                if z == 0:
                    continue
                self.update_pop_z(z)
                _ell_ = self.pop.get_ell(z)
                ps_2h_z = self.pop.get_conti_power_2h(zbin,wl,\
                                                      continuum_model,\
                                                      **kwargs)
                conti_ps_mtx[j,:] += np.interp(ell,_ell_,ps_2h_z)
            ps_shot_wl = continuum_model.Clsh(wl,**kwargs)
            conti_ps_mtx[j,:] += ps_shot_wl

        conti_ps_mtx = self.__convert_output(conti_ps_mtx,ell,output=output)

        return conti_ps_mtx

    def get_conti_xcorr_tot(self,ell,channels,zbins,continuum_model,\
                            output='cell',**kwargs):
        conti_xcorr_mtx = np.zeros((channels.shape[0],zbins.shape[0],ell.size))

        zbins_c = np.mean(zbins,axis=1)
        wls = np.mean(channels,axis=1)
        for j, zbin in enumerate(zbins):
            z = zbins_c[j]
            if z == 0:
                continue
            self.update_pop_z(z)
            _ell_ = self.pop.get_ell(z)
            n, b = self.pop.get_target_prop(zbin,None,None)
            for i, wl in enumerate(wls):
                ps = self.pop.get_conti_xcorr_chan(zbin,wl,continuum_model,\
                                                   n_target=n,b_target=b,\
                                                   **kwargs)
                conti_xcorr_mtx[i,j,:] = np.interp(ell,_ell_,ps)

        conti_xcorr_mtx = self.__convert_output(conti_xcorr_mtx,ell,\
                                                output=output)
        
        return conti_xcorr_mtx
    
    def get_gal_ps_tot(self,ell,zbins,output='cell'):
        """
        Compute the auto power spectrum of the target galaxies for all
        redshift bins at once.
        """
        gal_ps_mtx = np.zeros((zbins.shape[0],ell.size))
        zbins_c = np.mean(zbins,axis=1)
        for i, zbin in enumerate(zbins):
            z = zbins_c[i]
            if z == 0:
                continue
            self.update_pop_z(z)
            _ell_ = self.pop.get_ell(z)
            n, b = self.pop.get_target_prop(zbin,None,None)
            ps = self.pop.get_gal_ps_chan(zbin,n_target=n,b_target=b)
            gal_ps_mtx[i,:] = np.interp(ell,_ell_,ps)

        gal_ps_mtx = self.__convert_output(gal_ps_mtx,ell,output=output)
        
        return gal_ps_mtx

    def get_xcorr_tot(self,ell,channels,zbins,xcorr_mask=None,\
                      sum_lines=False,output='cell'):
        """
        Compute the galaxy/EBL cross spectrum for the entire matrix at once.
        """
        xcorr_mtx = np.zeros((len(self.include_lines),channels.shape[0],\
                              zbins.shape[0],ell.size))

        mask_shape = (channels.shape[0],zbins.shape[0])
        if xcorr_mask is None:
            xcorr_mask = np.ones(mask_shape)
        else:
            pass
           # assert xcorr_mask.shape == mask_shape, 'Incompatible mask shape'

        zbins_c = np.mean(zbins,axis=1)
        for j, zbin in enumerate(zbins):

            if np.all(xcorr_mask[:,j] == 0):
                continue

            zbin_c = zbins_c[j]
            self.update_pop_z(zbin_c)

            _ell_ = self.pop.get_ell(np.mean(zbin))
            n, b = self.pop.get_target_prop(zbin,None,None)

            for i, chan in enumerate(channels):
                for k, line in enumerate(self.include_lines):
                    ps = self.pop.get_xcorr_chan_line(chan,zbin=zbin,\
                                                      n_target=n,b_target=b,\
                                                      line=line)
                    xcorr_mtx[k,i,j,:] += np.interp(ell,_ell_,ps)

        if sum_lines: # sum over all lines
            xcorr_mtx = xcorr_mtx.sum(axis=0)

        xcorr_mtx = self.__convert_output(xcorr_mtx,ell,output=output)

        return xcorr_mtx

    def get_ps_tot(self,ell,channels,sum_lines=False,output='cell'):
        """
        Compute the EBL auto power spectrum for the entire matrix at once.
        """
        ps_mtx = np.zeros((len(self.include_lines),channels.shape[0],\
                           channels.shape[0],ell.size))

        for k, line in enumerate(self.include_lines):

            # Redshift bins for this particular line
            # lam_obs = lam_rest * (1. + z)
            z_edges, z_centers = mt.get_channel_zbin(channels,line)
            
            for j, chan1 in enumerate(channels):
                zbin = z_edges[j]
                z = z_centers[j]
                if z == 0:
                    continue
                self.update_pop_z(z)
                _ell_ = self.pop.get_ell(z)
                for i, chan2 in enumerate(channels):
                    if i != j:
                        continue
                    ps = self.pop.get_ps_chan_line(chan1,zbin=zbin,\
                                                   line=line)
                    ps_mtx[k,j,i,:] += np.interp(ell, _ell_, ps)

        if sum_lines: # sum over all lines
            ps_mtx = ps_mtx.sum(axis=0)

        ps_mtx = self.__convert_output(ps_mtx,ell,output=output)

        return ps_mtx

    def get_ps_chan(self,ell,channel):
        """
        Compute the EBL auto power spectrum for a particular spectral channel.
        """
        ps_mtx = np.zeros((len(self.include_lines), ell.size))

        for k, line in enumerate(self.include_lines):
            # Redshift bins for this particular line
            # lam_obs = lam_rest * (1. + z)
            zbin, z = mt.get_channel_zbin(channel,line)
            if z == 0:
                continue
            self.update_pop_z(z)
            _ell_ = self.pop.get_ell(z)
            ps = self.pop.get_ps_chan_line(channel,zbin=zbin,line=line)
            ps_mtx[k,:] += np.interp(ell,_ell_,ps)

        return ps_mtx

    def get_ebl_ps(self,scales,waves,waves2=None,zbins=None,xcorr_mask=None,\
                   wave_units='mic',scale_units='ell',flux_units='si',\
                   **kwargs):
        """
        Get EBL power spectrum.

        Parameters
        ----------
        scales : np.ndarray
            ell modes
        waves : np.ndarray
            Spectral channel definitions, with shape (n_channels,2)
        waves2 : np.ndarray
            Spectral channel definitions of 2nd channel, in the case of
            internal crosses. If defined, should have the same dimensions as
            `waves`. Defaults to `None` and currently not supported.
        zbins : np.ndarray
            Redshift bins, if computing galaxy cross spectra. Defaults to
            `None` (internal autos or crosses).
        xcorr_mask : np.ndarray
            Mask for galaxy cross spectra.

        Returns
        -------
        ps_tot : np.ndarray
            Total EBL power spectrum, with shape (len(scales),len(waves))
        """
        if zbins is None: # Internal autos or crosses
            # note that only autos are computed ...
            # This is a 3D array of shape (waves,waves,scales)
            ps_tot = self.get_ps_tot(scales,waves,sum_lines=True)

            # Autos -- only want diag. elements, with shape (scales, waves)
            ps_tot = ps_tot.diagonal()
        else: # Galaxy crosses
            # This is a 3D array of shape (waves,zbins,scales)
            ps_tot = self.get_xcorr_tot(scales,waves,zbins,\
                                        xcorr_mask=xcorr_mask,sum_lines=True)

        return ps_tot
