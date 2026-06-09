import numpy as np

##
# On to function definitions
def kwargs_to_args(kw, params):
    """
    Take a set of keywords arguments used to run a model and
    convert them to a list of arguments that emcee will understand.
    """
    args = []
    for i, par in enumerate(params):
        args.append(kw[par])

    return args

def args_to_kwargs(args, params):
    kwargs = {}

    # Update kwargs with current walker position.
    for i, arg in enumerate(args):
        kwargs[params[i]] = arg

    return kwargs

def lnprior(x, *args):
    """ Assess the prior. """

    params, priors = args

    kwargs = args_to_kwargs(x, params)

    if 'slope_lo_sfr' and 'slope_hi_sfr' in params:
        if kwargs['slope_lo_sfr'] > kwargs['slope_hi_sfr']:
            return -np.inf
    
    for key in kwargs:
        if priors[key][0] <= kwargs[key] < priors[key][1]:
            continue

        return -np.inf

    return 0.0

def lnpost(x, *args):
    """
    Compute posterior probability for model described by given parameters.
    """

    data, model, params, priors = args

    # First, retrieve our prior, and return if we're in violation of it.
    lnP = lnprior(x, params, priors)

    if not np.isfinite(lnP):
        return -np.inf

    # Next, convert our list of parameters to a dictionary that we can
    # pass to our modeling class.
    kwargs = args_to_kwargs(x, params)
    
    model.update(**kwargs)

    # So far, our only choice is whether Ha, Hb, or both, are included in the
    # fit, hence the if statements below.

    lnL = 0.0

    ##
    # Fitting cross-correlations
    if 'xcorr' in data:
        assert model.target_prop is not None 

        # Recall that `xcorr` will be (channels, z bins, ell modes)

        lbins = data['ellbins']
        channels = data['channels']
        zbins = data['zbins']

        # Can include series of channels in fit, though the input model
        # must be a ToyEmissionLineUniverse rather than ToyGalaxyPopulation.
        # The latter will not have a `get_xcorr_mtx` method.
        if 'fit_mask' in data:
            # Just ones and zeros.
            xcorr_mask = data['fit_mask']
            
            D_ell = model.get_xcorr_tot(lbins,channels,zbins,\
                                        xcorr_mask=xcorr_mask,sum_lines=True,\
                                        output='dell')
            ps_mod = np.ma.array(D_ell, mask=np.logical_not(xcorr_mask))

            ps_dat = data['xcorr']
            ps_err = data['xcorr_err']

            _lnL = (ps_dat - ps_mod)**2 / ps_err**2
            lnL += -0.5 * np.sum(_lnL)
            
        else:
            assert 'fit_channels' in data, \
                "Must supply list of channel IDs to be included in fit!"

            line_ids = data['fit_lines']
            chan_ids = data['fit_channels']
            zbin_ids = data['fit_zbins']

            chan = channels[chan_ids[0]]
            zbin = zbins[zbin_ids[0]]
            
            ps_dat = data['xcorr'][chan_ids[0], zbin_ids[0]]
            ps_err = data['xcorr_err'][chan_ids[0], zbin_ids[0]]
            
            _ps_mod = model.get_xcorr_chan_line(chan,zbin=zbin) # line?
            
            lmod = model.get_ell(np.mean(zbin))

            # Need to interpolate back onto right ell bins
            ps_mod = np.interp(lbins, lmod, _ps_mod)

            lnL += -0.5 * np.sum((ps_dat - ps_mod)**2 / ps_err**2)

    if 'fit_lf_lines' in data:
        fit_lf_lines = data['fit_lf_lines']
        lfs = data['line_lfs']
        for line in fit_lf_lines:
            if line == 'HbOIII':
                lines_use = ['Hb','OIII_5007','OIII_4959']
            elif line == 'HaNII':
                lines_use = ['Ha','NII']
            else:
                lines_use = line
            lf_l = lfs[line]
            zs_l = list(lf_l.keys())
            for z_l in zs_l:
                lf_l_z = lf_l[z_l]
                logL, phi = model.get_lf_line(z=z_l,line=lines_use)
                ymod_lf_l_z = np.interp(lf_l_z['log10L'],logL,np.log10(phi))
                _lnL = -0.5 * (lf_l_z['phi'] - ymod_lf_l_z)**2 / \
                    lf_l_z['phi_err']**2
                lnL += np.sum(_lnL)

                if np.any(np.isnan(_lnL)):
                    return -np.inf
    
    ##
    # Commenting this check out as it shouldn't be necessary.
    # If we get NaNs, should accept error and troubleshoot from there.
    #if np.any(np.isnan(lnL)) or np.any(np.isnan(lnP)):
    #    return -np.inf

    ##
    # Done!
    return lnL + lnP
