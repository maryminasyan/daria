import numpy as np
from ..utils.line_info import line_rwaves

def set_attrs(obj,default_dict,overwrite=False,**kwargs):
    """
    Set attributes for some object `obj`.

    Parameters
    ----------
    obj : Python class instance
        Class to which you want to assign attributes
    default_dict : dictionary
        Default values for all the attributes you're setting.
    overwrite : bool
        If `obj` already has some value for an attribute defined as an item in
        `default_dict`, overwrite the value in `obj` with the value in
        `default_dict`. If False, only assign `default_dict` items if they
        are not already attributes of `obj`. Defaults to False.
    **kwargs : dictionary of keyword arguments
        Keyword arguments you want to set as attributes of `obj`. Can also
        use to redefine attributes of kwargs; see `overwrite` above for what
        happens to attributes that are already defined in `obj` but not
        in `kwargs`.
    """
    default_keys = default_dict.keys()
    given_keys = kwargs.keys()

    # Set relevant provided attrs
    for given_key in given_keys:
        if given_key in default_keys:
            setattr(obj,given_key,kwargs[given_key])

    # Set relevant unprovided attrs
    for default_key in default_keys:
        do_overwrite = overwrite and (default_key not in given_keys)
        if do_overwrite or not hasattr(obj,default_key):
            setattr(obj,default_key,default_dict[default_key])

def get_mlim(mlim):
    """
    Get integration bound corresponding to `mlim`, the halo mass
    corresponding to the limiting magnitude of a population of target galaxies.

    If `mlim` is None, integrate to arbitrarily small halo masses (really, the
    limit defined upon initialization of `hmf`). If `mlim` is not None, it
    means we're computing the cross-shot term, i.e., we only want to know the
    emissivity from the target galaxies.
    """
    if mlim is None:
        return 0
    else:
        return mlim

def get_mask(mask):
    """
    Get integration bound corresponding to `mask`, the halo mass
    corresponding to the masking depth.

    If `mask=None`, assume no mask, in which case integrate to arbitrarily
    large halo masses (really, the upper limit defined upon init of `hmf`).
    """
    if mask is None:
        return np.inf
    else:
        return mask

def get_metal_line_model(metal_line_model):
    """
    Get `metal_line_model` per user input. This prevents any potential
    downstream issues.
    """
    if not metal_line_model:
        return False
    else:
        if isinstance(metal_line_model,str):
            lower = metal_line_model.lower()
            # names could use some workshopping?
            assert lower in ['l_sfr','indep_pow','o3_dep_n2'],\
                'Unsupported metal line model'
            return lower
        else:
            return 'l_sfr'

def get_default_dict(metal_line_model):
    """
    Get dictionary of default values for parameters corresponding to the
    user-selected metal line model. Note that the `'l_sfr'` model has no
    further parametrization, hence no default parameters assigned here.
    """
    default_dict = {}

    if metal_line_model == 'indep_pow':
        default_dict['N2_norm'] = -1.5
        default_dict['N2_slope'] = 0.5
        default_dict['O3_norm'] = 0.5 # O3 defaults are somewhat arbitrary
        default_dict['O3_slope'] = -0.5
    elif metal_line_model == 'o3_dep_n2':
        default_dict['N2_norm'] = -1.5
        default_dict['N2_slope'] = 0.5
        default_dict['O3_const'] = 1.1
        default_dict['O3_num'] = 0.61
        default_dict['O3_denom'] = 0.08

    return default_dict

def get_metal_line_model_keys(metal_line_model):
    default_dict = get_default_dict(metal_line_model)
    return default_dict.keys()

def get_metal_line_kwargs(obj,metal_line_model):
    kwargs = {}
    keys = get_metal_line_model_keys(metal_line_model)
    for key in keys:
        kwargs[key] = getattr(obj,key)
    return kwargs

def get_channel_zbin(channel,line):
    return channel * 1e4 / line_rwaves[line] - 1
