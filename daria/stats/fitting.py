import numpy as np

##
# On to function definitions
def kwargs_to_args(kw, params, is_log):
    """
    Take a set of keywords arguments used to run a model and
    convert them to a list of arguments that emcee will understand.
    """
    args = []
    for i, par in enumerate(params):
        if par in is_log:
            args.append(np.log10(kw[par]))
        else:
            args.append(kw[par])

    return args

def args_to_kwargs(args, params, is_log):
    kwargs = {}

    # Update kwargs with current walker position.
    for i, arg in enumerate(args):
        par = params[i]

        if par in is_log:
            kwargs[par] = 10**arg
        else:
            kwargs[par] = arg

    return kwargs

def lnprior(x, *args):
    """ Assess the prior. """

    params, is_log, priors = args

    kwargs = args_to_kwargs(x, params, is_log)

    for key in kwargs:
        if priors[key][0] <= kwargs[key] < priors[key][1]:
            continue

        return -np.inf

    return 0.0

def lnpost(x, *args):
    """
    Compute posterior probability for model described by given parameters.
    """

    data, model, params, is_log, priors = args

    # First, retrieve our prior, and return if we're in violation of it.
    lnP = lnprior(x, params, is_log, priors)

    if not np.isfinite(lnP):
        return -np.inf

    # Next, convert our list of parameters to a dictionary that we can
    # pass to our modeling class.
    kwargs = args_to_kwargs(x, params, is_log)
    
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

       # kw = base_kwargs.copy()
       # kw.update(kwargs)

        # Can include series of channels in fit, though the input model
        # must be a ToyEmissionLineUniverse rather than ToyGalaxyPopulation.
        # The latter will not have a `get_xcorr_mtx` method.
        if 'fit_mask' in data:
            # Just ones and zeros.
            xcorr_mask = data['fit_mask']
            
            C_ell = model.get_xcorr_tot(lbins,channels,zbins,\
                                        xcorr_mask=xcorr_mask,sum_lines=True)
            ells = lbins[None,None,:]
            D_ell = ells * (ells + 1) * C_ell / 2. / np.pi
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

    ##
    # Line LFs
    if 'lf_Ha' in data:
        z_Ha = data['lf_Ha_z']
        logL_Ha = data['lf_Ha_logL']
        for i, z in enumerate(z_Ha):
            logL, phi = model.get_lf_line(z=z, line='Ha')
            ymod_lf = np.interp(logL_Ha, logL, np.log10(phi))
            _lnL = -0.5 * (data['lf_Ha'][i,:] - ymod_lf)**2 \
                / data['lf_Ha_err'][i,:]**2

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
