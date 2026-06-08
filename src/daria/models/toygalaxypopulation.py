import numpy as np
from types import FunctionType
from functools import cached_property
from hmf import MassFunction
from dust_attenuation.averages import C00
from ..utils.line_info import line_rwaves, line_lpersfr
from ..utils.constants import c, m_per_mpc, erg_per_s_per_nW, sqdeg_per_std
from . import model_tools as mt

class ToyGalaxyPopulation(object):
    def __init__(self,target_prop=None,metal_line_model='l_sfr',mlim=11,\
                 mask=12,mmin=8,rturn=1,norm_sfr=-4,mbreak_sfr=12,\
                 slope_lo_sfr=1.5,slope_hi_sfr=0.5,norm_Av=1.,slope_Av=0.1,\
                 **kwargs):
        """
        Note - mlim, mask, mmin, and norm_sfr are all LOG quantities.
        
        Initialize a toy model capable of (for now) doing cross-correlations
        between a galaxy sample and intensity map. For more information on
        what kwargs to provide for different metal_line_model inputs, see
        the `ToyEmissionLineUniverse` init docstring.
        """
        self.target_prop = target_prop
        self.metal_line_model = mt.get_metal_line_model(metal_line_model)
        self.mlim = mt.get_mlim(mlim)
        self.mask = mt.get_mask(mask)

        # Model parameters
        self.mmin = mmin
        self.rturn = rturn
        self.norm_sfr = norm_sfr
        self.mbreak_sfr = mbreak_sfr
        mt.set_slope_sfr(self,slope_lo_sfr,slope_hi_sfr) # bookkeeping
        self.norm_Av = norm_Av
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
                                    'mlim','mask','kwargs']}

        # Add individual kwargs entries; assume user defined them correctly...
        init_args.update(kwargs)
        self.init_args = init_args

    def update(self,**kwargs):
        """
        It is expensive to continually re-initialize this class during
        likelihood evaluation. You can avoid this cost by initializing once
        per thread and overwriting arguments as needed. Make sure to initialize
        properly to avoid messy errors from potential inconsistencies, etc.
        """
        mt.set_attrs(self,self.init_args,overwrite=False,**kwargs)
        mt.set_slope_sfr(self,self.slope_lo_sfr,self.slope_hi_sfr)

    def exponentiate(self,log_quantity):
        return 10**log_quantity
        
    def get_integration_bounds(self,cross_shot=False):
        m = self.m
        if cross_shot:
            ''' For the cross-shot term, we only want to know the emissivity
            from the target galaxies, so we only integrate down to the
            limiting magnitude of that population. '''
            mlo = self.exponentiate(self.mlim)
        else:
            mlo = self.exponentiate(self.mmin)
        mhi = self.exponentiate(self.mask)
        ok = np.logical_and(m >= mlo, m < mhi)
        return m[ok], ok
    
    @property
    def hmf(self):
        if not hasattr(self, '_hmf'):
            hmf = MassFunction(z=0,Mmin=4,Mmax=16,dlog10m=0.01,\
                               hmf_model='Tinker10')
            self._hmf = hmf
        return self._hmf

    @cached_property
    def h(self):
        return self.hmf.cosmo.h

    @cached_property
    def m(self):
        return self.hmf.m / self.h

    @cached_property
    def k(self):
        return self.hmf.k * self.h

    @property
    def _dust(self):
        if not hasattr(self, '_dust_'):
          #  assert have_dustatt, "Need dust_attenuation package for this!"
            self._dust_ = C00
        return self._dust_

    @cached_property
    def tab_dust_Av(self):
        return np.arange(0, 10.1, 0.1)

    @cached_property
    def tab_dust_waves_c(self):
        CC = self._dust(Av=1) # just need an instance to get wavelengths
        return np.arange(1e4 * CC.x_range[0], 1e4 * CC.x_range[1])

    @cached_property
    def tab_dust_waves_e(self):
        return self.tab_dust_waves_c - 0.5

    @cached_property
    def tab_dust_attenuation(self):
        tab_C00 = np.zeros((self.tab_dust_waves_c.size,self.tab_dust_Av.size))
        for i, Av in enumerate(self.tab_dust_Av):
            C00 = self._dust(Av=Av)
            tab_C00[:,i] = C00(self.tab_dust_waves_c * 1e-4)
        return tab_C00

    def get_dust_attenuation(self,m,line='Ha'):
        Av = self.norm_Av * (m / 1e10)**self.slope_Av
        line_w = line_rwaves[line]
        iline = np.digitize(line_w, self.tab_dust_waves_e) - 1
        return np.interp(Av,self.tab_dust_Av,\
                         self.tab_dust_attenuation[iline,:])

    def get_dust_transmission(self,m,line='Ha'):
        return 10**(-self.get_dust_attenuation(m,line=line) / 2.5)

    def get_halo_bias(self,z):
        """
        Return the linear bias of dark matter halos at redshift `z`.

        See `self.m` for the corresponding halo masses [Msun].
        """
        self.hmf.update(z=z)

        g = self.hmf.growth_factor
        sigma = self.hmf._sigma_0

        # Note also that this is also HMF's definition of nu
        delta_sc = 1.686
        nu = (delta_sc / sigma / g)

        y = np.log10(200.)
        A = 1. + 0.24 * y * np.exp(-(4. / y)**4)
        a = 0.44 * y - 0.88
        B = 0.183
        b = 1.5
        C = 0.019 + 0.107 * y + 0.19 * np.exp(-(4. / y)**4)
        c = 2.4

        bias = 1. - A * (nu**a / (nu**a + delta_sc**a)) \
             + B * nu**b + C * nu**c

        return bias

    def get_sfr(self,m):
        """
        Return the SFR (in Msun/yr) of halos of masses `m`, under the
        assumption of a double power-law relationship.

        Parameters
        ----------
        m : int, float, np.ndarray
            Halo mass(es) of interest [Msun]
        """
        mbreak_sfr = self.exponentiate(self.mbreak_sfr)
        norm_sfr = self.exponentiate(self.norm_sfr)
        
        normcorr = (1e10 / mbreak_sfr)**-self.slope_lo_sfr + \
            (1e10 / mbreak_sfr)**-self.slope_hi_sfr

        return normcorr * norm_sfr / \
            ((m / mbreak_sfr)**-self.slope_lo_sfr + \
             (m / mbreak_sfr)**-self.slope_hi_sfr)

    def get_focc(self,m):
        mmin = self.exponentiate(self.mmin)
        return 1 - np.exp(-(m / mmin)**self.rturn)

    def get_sfrd(self,z):
        """
        Returns cosmic star-formation rate density (Msun/yr/cMpc^3) at `z`.
        """
        self.hmf.update(z=z)
        m = self.m
        sfr = self.get_sfr(m)
        focc = self.get_focc(m)
        dndlnm = self.hmf.dndlnm * self.h**3
        sfrd = np.trapz(dndlnm * sfr * focc, x=np.log(m))
        return sfrd

    def get_lum_line(self,m,line='Ha'):
        """
        Get the line luminosity (reddened by dust) for given `line`.
        """
        sfr = self.get_sfr(m)
        Tdust = self.get_dust_transmission(m,line=line)

        is_H_line = line.lower().startswith('h') or line in ['Pa','Pab']
        metal_line_model = self.metal_line_model

        if not metal_line_model and not is_H_line:
            return np.zeros_like(sfr)
        elif (metal_line_model == 'l_sfr') or is_H_line:
            return sfr * line_lpersfr[line] * Tdust
        else:
            # If we're here, we're doing some kind of BPT+ modeling.
            # First, retrieve *intrinsic* H-a and H-b line emission as
            # we'll scale from there.
            LHa = sfr * line_lpersfr['Ha']
            LHb = sfr * line_lpersfr['Hb']

            N2 = self.N2_norm + self.N2_slope * np.log10(m / 1e10)
            if line == 'NII':
                return LHa * 10**N2 * Tdust
            elif line.startswith('OII'):
                if metal_line_model == 'o3_dep_n2':
                    O3 = self.O3_const + (self.O3_num / (N2 + self.O3_denom))
                   # O3 = 1.1 + 0.61 / (N2 + 0.08)
                elif metal_line_model == 'indep_pow':
                    O3 = self.O3_norm + self.O3_slope * np.log10(m / 1e10)

                if line == 'OIII_5007':
                    return LHb * 10**O3 * Tdust
                elif line == 'OIII_4959':
                    return LHb * 10**O3 * Tdust / 3.
                elif line == 'OII':
                    # For now, just fixed conversion from OIII
                    LOIII = (4. / 3.) * LHb * 10**O3 * Tdust
                    return LOIII * line_lpersfr['OII'] / line_lpersfr['OIII']
                else:
                    raise NotImplemented('shouldnt do total OIII!')
            else:
                return sfr * line_lpersfr[line] * Tdust

    def get_lf_line(self,z,line='Ha'):
        """
        Compute the line luminosity function at redshift `z`.
        """
        self.hmf.update(z=z)
        m = self.m

        log10lum = np.log10(self.get_lum_line(m,line=line))
        dlog10lum = np.diff(log10lum)

        ''' The relationship between line luminosity and halo mass can be
        double- or even triple-valued depending on the strength of dust
        attenuation. If it is monotonic, we have a straightforward mapping
        from the halo mass function to the line luminosity function. Otherwise,
        we must handle the luminosity function more carefully by splitting
        it into chunks where the line luminosity vs. halo mass relation
        increases/decreases. '''
        if np.all(dlog10lum > 0):
            # monotonically increasing; ideal case
            dlnmdlog10l = np.diff(np.log(m)) / dlog10lum
            focc = self.get_focc(m[0:-1])
            dndlnm = self.hmf.dndlnm[0:-1] * self.h**3
            lf = dndlnm * dlnmdlog10l * focc
            return log10lum[0:-1], lf
        else:
            # double/triple-valued behavior
            decreasing_idxs = np.where(dlog10lum < 0)[0]

            ''' Sometimes you may run into models where luminosity decreases
            as halo mass increases, starting from the smallest halos. For
            now I am intervening with a workaround to avoid/penalize these
            models when running inference. '''
            if decreasing_idxs[0] == 0:
                return np.array([-np.inf,np.inf]), \
                    np.array([-np.inf,-np.inf])
            
            triple_valued = np.max(decreasing_idxs) < (len(log10lum)-2)

            ''' Compute the luminosity function in each chunk. Add regions
            across chunks with overlapping luminosities to get all the halos
            contributing to those bins. '''
            chunk1_idxs = np.arange(0,np.min(decreasing_idxs)+1)
            chunk2_idxs = np.arange(np.min(decreasing_idxs),\
                                    np.max(decreasing_idxs)+2)
            chunks = [chunk1_idxs,chunk2_idxs]
            if triple_valued:
                chunk3_idxs = np.arange(np.max(decreasing_idxs)+1,\
                                        len(log10lum))
                chunks.append(chunk3_idxs)
                
            focc = self.get_focc(m)
            dndlnm = self.hmf.dndlnm * focc * self.h**3
            dlnm = np.diff(np.log(m))[0] # even spacing
            dn = dndlnm * dlnm

            def get_dn_at_lnm(lnm):
                return np.interp(lnm,np.log(m),dn)

            log10lum_fin = np.linspace(np.min(log10lum),
                                       np.max(log10lum),
                                       num=2000)
            dn_fin = np.zeros_like(log10lum_fin)
            
            for chunk_idxs in chunks:
                lnm_chunk = np.log(m[chunk_idxs])
                log10lum_chunk = log10lum[chunk_idxs]
                dn_chunk = get_dn_at_lnm(lnm_chunk)
                increasing = ((log10lum_chunk[1] - log10lum_chunk[0]) > 0)
                
                def get_lnm_at_log10lum(log10l):
                    if increasing:
                        return np.interp(log10l,log10lum_chunk,lnm_chunk)
                    else:
                        ''' np interp handles monotonically increasing
                        functions. So if luminosity decreases with halo mass,
                        flip the relation and then flip the interpolated
                        result again for the correct lnm. '''
                        return np.flip(np.interp(log10l,\
                                                 np.flip(log10lum_chunk),\
                                                 lnm_chunk))
                
                fin_idxs = np.logical_and(log10lum_fin >= \
                                          np.min(log10lum_chunk),\
                                          log10lum_fin <= \
                                          np.max(log10lum_chunk))
                                                                 
                log10lum_chunk_fin = log10lum_fin[fin_idxs]
                lnm_chunk_fin = get_lnm_at_log10lum(log10lum_chunk_fin)
                dn_fin[fin_idxs] += get_dn_at_lnm(lnm_chunk_fin)

            dlog10lum_fin = np.diff(log10lum_fin)[0] # even spacing
            dndlog10lum_fin = dn_fin / dlog10lum_fin
            return log10lum_fin, dndlog10lum_fin

    def get_log_lum_ratio(self,m,line_top,line_bot):
        """
        Get the (log) line luminosity ratio of `line_top`/`line_bot` for
        halos of masses `m`.
        """
        def get_lum(line):
            return self.get_lum_line(m,line=line)

        Ltop = get_lum(line_top)
        Lbot = get_lum(line_bot)
        return np.log10(Ltop/Lbot)

    def get_bpt(self,m,line_xtop='NII',line_xbot='Ha',line_ytop='OIII_5007',\
                line_ybot='Hb'):
        """
        Get `line_xtop`/`line_xbot` and `line_ytop`/`line_ybot` (log) line
        luminosity ratios for halos of masses `m`.

        Default line ratios are [NII]/Ha (x) and [OIII(5007)]/Hb (y).
        """
        x = self.get_log_lum_ratio(m,line_top=line_xtop,line_bot=line_xbot)
        y = self.get_log_lum_ratio(m,line_top=line_ytop,line_bot=line_ybot)
        return x, y

    def get_freq_line(self,line):
        return c * 1e8 / line_rwaves[line]

    def get_bias_line(self,z,line='Ha'):
        """
        Compute the line-intensity-weighted bias at redshift `z`.
        """
        self.hmf.update(z=z)
        m_use, ok = self.get_integration_bounds()

        bh = self.get_halo_bias(z)[ok]
        lum = self.get_lum_line(m_use,line=line)
        focc = self.get_focc(m_use)
        dndlnm = self.hmf.dndlnm[ok] * self.h**3

        top = np.trapz(dndlnm * focc * lum * bh, x=np.log(m_use))
        bot = np.trapz(dndlnm * focc * lum, x=np.log(m_use))

        return top / bot

    def get_nuInu(self,z,line='Ha',cross_shot=False):
        """
        Compute the line emissivity (nW/m^2/sr) at redshift `z`.
        """
        self.hmf.update(z=z)
        m_use, ok = self.get_integration_bounds(cross_shot=cross_shot)

        H, chi = self.get_Hcm_and_chi(z)
        nu = self.get_freq_line(line)                           # Hz
        dchi_dnu = c * (1. + z)**2 / H / nu                     # Mpc Hz^-1
        lum = self.get_lum_line(m_use,line)                     # erg/s
        d_L = self.hmf.cosmo.luminosity_distance(z).value       # Mpc
        d_A = self.hmf.cosmo.angular_diameter_distance(z).value # Mpc

        norm = (nu / 4. / np.pi) * dchi_dnu * d_A**2 / d_L**2   # Mpc sr^-1

        focc = self.get_focc(m_use)
        dndlnm = self.hmf.dndlnm[ok] * self.h**3

        integ = np.trapz(dndlnm * focc * lum, x=np.log(m_use))  # erg/s/Mpc^3

        return norm * integ / m_per_mpc**2 / erg_per_s_per_nW

    def get_z_dz(self,channel,line,zbin=None):
        """
        Figure out the appropriate "delta z" factors for a given line.

        Parameters
        ----------
        channel : np.ndarray
            Two-element array containing the spectral channel bin edges
            [micron].
        line : str
            Name of emission line of interest, e.g., 'Ha', 'Hb', 'OIII', etc.
        zbin : np.ndarray
            Two-element array containing the redshift bin edges. If this is
            not supplied, we'll assume a redshift bin that is a perfect match
            to the spectral channel edges.

        Returns
        -------
        Tuple containing (mean redshift probed, dz for galaxies, dz for line,
            combination "dz^{gl}" which is the delta z interval shared by
            galaxies and line)
        """
        # Compute redshift interval for this line and spectral channel
        zbin_l, z = mt.get_channel_zbin(channel,line)
        zlo_l, zhi_l = zbin_l
        dz_l = zhi_l - zlo_l

        # If user supplied a redshift bin, use it.
        # Note that the user may supply a redshift bin that our spectral line
        # of choice doesn't intersect in this channel. Check that first.
        if zbin is not None:
            if (zhi_l < zbin[0]) or (zlo_l > zbin[1]):
                dz_g = dz_gl = 0
                z = -np.inf
            else:
                dz_g = zbin[1] - zbin[0]
                z = np.mean(zbin)
                dz_gl = min(dz_l,dz_g)

        # Otherwise, assume "perfect" redshift bin that matches exactly
        # the range probed by the channel.
        # Compute relevant redshift intervals
        else:
            dz_gl = dz_g = dz_l

        return z, dz_g, dz_l, dz_gl

    def get_ell(self, z):
        self.hmf.update(z=z)
        return self.hmf.k * self.h * \
            self.hmf.cosmo.comoving_distance(z).value - 0.5

    def get_projected_volume(self, z, dz):
        """
        Returns the co-moving volume [in cMpc^3] of a 1 deg^2 'chunk' of the
        Universe centered at z +/ 0.5 * dz.
        """
        dVdzdsr = self.hmf.cosmo.differential_comoving_volume(z).value
        return dVdzdsr * dz / sqdeg_per_std

    def get_H_cm(self,z):
        """
        Hubble parameter in cm/s/Mpc (converted from km/s/Mpc)
        """
        return self.hmf.cosmo.H(z).value * 1e5

    def get_chi(self,z):
        return self.hmf.cosmo.comoving_distance(z).value

    def get_Hcm_and_chi(self,z):
        H = self.get_H_cm(z)
        chi = self.get_chi(z)
        return H, chi

    def get_pmm(self,z):
        """
        Return matter power spectrum P(k) at input `z`. The corresponding
        k modes are stored in `self.hmf.k`.
        """
        self.hmf.update(z=z)
        return self.hmf.power / self.h**3

    def get_conti_xcorr_2h(self,zbin,wl,continuum_model,b_target,**kwargs):
        z = np.mean(zbin)
        if z == 0:
            return np.zeros_like(self.hmf.k)
        else:
            H, chi = self.get_Hcm_and_chi(z)
            pmm = self.get_pmm(z)
            dz = zbin[1] - zbin[0]
            dnuInu_dz = continuum_model.dnuInu_dz(z,wl,**kwargs)
            bI = continuum_model.bI(z)
            return (H / c / chi**2) * b_target * bI * dnuInu_dz * pmm

    def get_conti_xcorr_shot(self,zbin,wl,continuum_model,n_target,**kwargs):
        z = np.mean(zbin)
        if z == 0:
            return 0
        else:
            dz = zbin[1] - zbin[0]
            dn_dz_dsr = n_target * sqdeg_per_std
            dnuInu_dz = continuum_model.dnuInu_dz(z,wl,**kwargs)
            return dz * dnuInu_dz / dn_dz_dsr

    def get_conti_xcorr_chan(self,zbin,wl,continuum_model,n_target=None,\
                             b_target=None,**kwargs):
        n_target,b_target = self.get_target_prop(zbin,n_target,b_target)
        ps_2h = self.get_conti_xcorr_2h(zbin,wl,continuum_model,b_target,\
                                        **kwargs)
        ps_sh = self.get_conti_xcorr_shot(zbin,wl,continuum_model,n_target,\
                                          **kwargs)
        return ps_2h + ps_sh
            
    def get_conti_power_2h(self,zbin,wl,continuum_model,**kwargs):
        """
        Auto 2-halo (linear) power of continuum-only EBL. User must supply
        a continuum model with the functions dnuInu_dz(z,wl,**kwargs) and
        bI(z).
        """
        z = np.mean(zbin)
        if z == 0:
            return np.zeros_like(self.hmf.k)
        else:
            H, chi = self.get_Hcm_and_chi(z)
            pmm = self.get_pmm(z)
            dz = zbin[1] - zbin[0]
            dnuInu_dz = continuum_model.dnuInu_dz(z,wl,**kwargs)
            bI = continuum_model.bI(z)
            return dz * (H / c / chi**2) * (dnuInu_dz * bI)**2 * pmm

    def get_gal_2h(self,zbin,b_target):
        """
        Auto 2-halo (linear) clustering of target galaxies.
        """
        z = np.mean(zbin)
        dz_g = zbin[1] - zbin[0]
        if dz_g == 0:
            return np.zeros_like(self.hmf.k)
        else:
            H, chi = self.get_Hcm_and_chi(z)
            pmm = self.get_pmm(z)
            return (H / chi**2 / dz_g / c) * b_target**2 * pmm

    def get_gal_shot(self,zbin,n_target):
        """
        Auto shot power of target galaxies.
        """
        z = np.mean(zbin)
        dz_g = zbin[1] - zbin[0]
        if dz_g == 0:
            return 0
        else:
            return 1/(dz_g * n_target * sqdeg_per_std)

    def get_gal_ps_chan(self,zbin,n_target=None,b_target=None):
        """
        Compute auto power of target galaxies.

        Parameters
        ----------
        zbin : np.ndarray
            Edges of redshift bin of interest
        n_target : int, float, None
            Surface density of galaxies in number per deg^2 (in this `zbin`).
        b_target : int, float, None
            Bias of galaxies in this redshift bin.
        """
        n_target, b_target = self.get_target_prop(zbin,n_target,b_target)

        ps_2h = self.get_gal_2h(zbin,b_target)
        ps_sh = self.get_gal_shot(zbin,n_target)

        return ps_sh + ps_2h
    
    def get_power_2h(self,channel,zbin=None,line='Ha'):
        """
        Compute auto EBL linear clustering (2-halo) power.

        Parameters
        ----------
        channel : np.ndarray
            Two-element array containing the spectral channel bin edges
            [micron].
        zbin : np.ndarray
            Two-element array containing the redshift bin edges. Defaults to
            `None`, in which case we assume a redshift bin that is a perfect
            match to the spectral channel edges.
        line : str
            Name of emission line of interest, e.g., 'Ha', 'Hb', 'OIII', etc.
        """
        z, dz_g, dz_l, dz_gl = self.get_z_dz(channel,line,zbin=zbin) # dz's
        
        if dz_g == 0:
            return np.zeros_like(self.hmf.k)
        else:
            H, chi = self.get_Hcm_and_chi(z)
            pmm = self.get_pmm(z)                   # matter power spectrum
            # line intensity weighted bias
            bl = self.get_bias_line(z, line=line)
            nuInu_l = self.get_nuInu(z, line=line)  # line intensity
            return (H * bl**2 / c / chi**2 / dz_l) * nuInu_l**2 * pmm

    def get_power_shot(self,channel,zbin=None,line='Ha'):
        """
        Compute auto EBL shot power.

        Parameters
        ----------
        channel : np.ndarray
            Two-element array containing the spectral channel bin edges
            [micron].
        zbin : np.ndarray
            Two-element array containing the redshift bin edges. Defaults to
            `None`, in which case we assume a redshift bin that is a perfect
            match to the spectral channel edges.
        line : str
            Name of emission line of interest, e.g., 'Ha', 'Hb', 'OIII', etc.
        """
        z, dz_g, dz_l, dz_gl = self.get_z_dz(channel, line, zbin=zbin)
        if dz_g == 0:
            return 0
        else:
            H, chi = self.get_Hcm_and_chi(z)
            nuInu_l = self.get_nuInu(z, line=line)
            return (1. / dz_l) * (H / chi**2 / c) * nuInu_l

    def get_ps_chan_line(self,channel,zbin=None,line='Ha'):
        """
        Compute auto EBL power, summing shot and linear regimes, for a single
        spectral channel and single emission line.

        Parameters
        ----------
        channel : np.ndarray
            Two-element array containing the spectral channel bin edges
            [micron].
        zbin : np.ndarray
            Two-element array containing the redshift bin edges. If this is
            not supplied, we'll assume a redshift bin that is a perfect match
            to the spectral channel edges.
        line : str
            Name of emission line of interest, e.g., 'Ha', 'Hb', 'OIII', etc.
        """
        def get_power(scales):
            if scales == '2h':
                func = self.get_power_2h
            else:
                assert scales == 'shot'
                func = self.get_power_shot
            return func(channel,zbin=zbin,line=line)

        ps_2h = get_power('2h')
        ps_sh = get_power('shot')

        return ps_sh + ps_2h

    def get_ps_chan(self,channel,zbins,include_lines=['Ha']):
        """
        Compute auto EBL power for a single channel, including contributions
        from potentially many different emission lines. Put another way, we're
        summing all contributions down a column (spanning all z bins) in our
        auto-correlation matrix.
        """
        z = np.mean(zbins, axis=1)
        dz = np.diff(zbins, axis=1)
        ps = np.zeros(zbins.shape[0])
        for line in include_lines:
            for j, zbin in enumerate(zbins):
                ps[j] += self.get_ps_chan_line(channel,zbin=zbin,line=line)
        return ps

    def get_xpower_2h(self,channel,n_target,b_target,zbin=None,line='Ha'):
        """
        Compute large-scale cross-power (nW^2 m^-4 sr^-2) between galaxies
        and EBL.

        Parameters
        ----------
        channel : np.ndarray
            Two-element array containing the spectral channel bin edges
            [micron].
        n_target : int, float
            Surface density of target galaxies [sr^-2].
        b_target : int, float
            Linear bias of target galaxies.
        zbin : np.ndarray
            Two-element array containing the redshift bin edges. Defaults to
            `None`, in which case we assume a redshift bin that is a perfect
            match to the spectral channel edges.
        line : str
            Name of emission line of interest, e.g., 'Ha', 'Hb', 'OIII', etc.
        """
        z, dz_g, dz_l, dz_gl = self.get_z_dz(channel, line, zbin=zbin) # dz's

        if (dz_g == 0) or (dz_l == 0):
            return np.zeros_like(self.hmf.k)

        H, chi = self.get_Hcm_and_chi(z)
        pmm = self.get_pmm(z)                   # matter power spectrum
        bl = self.get_bias_line(z, line=line)   # line intensity weighted bias
        nuInu_l = self.get_nuInu(z, line=line)  # line intensity

        return (H * dz_gl * b_target * bl / c / chi**2 / dz_g / dz_l) \
            * nuInu_l * pmm

    def get_xpower_shot(self,channel,n_target,zbin=None,line='Ha'):
        """
        Compute cross-shot-power between galaxies and EBL.

        Parameters
        ----------
        channel : np.ndarray
            Two-element array containing the spectral channel bin edges
            [micron].
        n_target : int, float
            Surface density of galaxies in target sample, i.e., galaxies we're
            cross correlating with [# / deg^2].
        zbin : np.ndarray
            Two-element array containing the redshift bin edges. If this is
            not supplied, we'll assume a redshift bin that is a perfect match
            to the spectral channel edges.
        line : str
            Name of emission line of interest, e.g., 'Ha', 'Hb', 'OIII', etc.
        """
        z, dz_g, dz_l, dz_gl = self.get_z_dz(channel,line,zbin=zbin)

        if (dz_g == 0) or (dz_l == 0):
            return 0

        nuInu_l = self.get_nuInu(z,line=line,cross_shot=True)

        return (dz_g / dz_l) * nuInu_l / (n_target * sqdeg_per_std)

    def get_xcorr_chan_line(self,channel,zbin=None,n_target=None,\
                            b_target=None,line='Ha'):
        """
        Compute cross power between galaxies and EBL, summing shot and linear
        regimes, for a single spectral channel and single emission line.

        Parameters
        ----------
        channel : np.ndarray
            Two-element array containing the spectral channel bin edges
            [micron].
        n_target : int, float
            Surface density of galaxies in target sample, i.e., galaxies we're
            cross correlating with [# / deg^2].
        b_target : int, float
            Linear bias of target galaxies.
        zbin : np.ndarray
            Two-element array containing the redshift bin edges. If this is
            not supplied, we'll assume a redshift bin that is a perfect match
            to the spectral channel edges.
        line : str
            Name of emission line of interest, e.g., 'Ha', 'Hb', 'OIII', etc.
        """
        n_target, b_target = self.get_target_prop(zbin,n_target,b_target)

        ps_2h = self.get_xpower_2h(channel,n_target,b_target,zbin=zbin,\
                                   line=line)
        ps_sh = self.get_xpower_shot(channel,n_target,zbin=zbin,line=line)

        return ps_sh + ps_2h

    def get_xcorr_chan(self,channel,zbins,n_target=None,b_target=None,\
                       include_lines=['Ha']):
        """
        Compute cross power for a single channel, including contributions from
        potentially many different emission lines. Put another way, we're
        summing all contributions down a column (spanning all z bins) in our
        cross-correlation matrix.

        See `get_xcorr_chan_line` docstring for info on input arguments.
        """
        z = np.mean(zbins, axis=1)
        dz = np.diff(zbins, axis=1)
        ps = np.zeros(zbins.shape[0])
        for line in include_lines:
            for j, zbin in enumerate(zbins):
                n, b = self.get_target_prop(zbin,n_target,b_target)
                ps[j] += self.get_xcorr_chan_line(channel,zbin=zbin,\
                                                  n_target=n,b_target=b,\
                                                  line=line)
        return ps

    def get_target_prop(self,zbin,n_target=None,b_target=None):
        """
        Return the key properties of the galaxy population we're cross-
        correlating with.

        .. note :: If `n_target` or `b_target` are None, will consult with the
            contents of `self.target_prop` set at initialization time.

        Parameters
        ----------
        zbin : tuple, list, np.ndarray
            Edges of redshift bin of interest (2-element list or tuple).
        n_target : int, float, None
            Surface density of galaxies in number per deg^2 (in this `zbin`).
        b_target : int, float, None
            Bias of galaxies in this redshift bin.
        """
        # First, check for None
        if n_target is None:
            if 'dndV' in self.target_prop:
                _n_target = lambda z, dz: self.target_prop['dndV'](z) \
                    * self.get_projected_volume(z, dz)
            else:
                assert 'dndOmega' in self.target_prop
                _n_target = self.target_prop['dndOmega']
        else:
            _n_target = n_target

        if b_target is None:
            _b_target = self.target_prop['bias']
        else:
            _b_target = b_target

        z = np.mean(zbin)
        dz = np.diff(zbin)

        # Now, handle functions vs numbers
        if type(_n_target) == FunctionType:
            _n = _n_target(z, dz)
        else:
            _n = _n_target

        if type(_b_target) == FunctionType:
            _b = _b_target(z)
        else:
            _b = _b_target

        return _n, _b
