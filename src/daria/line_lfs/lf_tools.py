import numpy as np

def correct_little_h(data,quantity,h_in=0.7,h_out=0.6766):
    if quantity == 'phi':
        return data + (3*np.log10(h_out/h_in))
    elif quantity == 'log10L':
        return data + (2*np.log10(h_in/h_out))
    else:
        return data

def replace_phi(model,data_dict,lines):
    for z in list(data_dict.keys()):
        log10L, phi = model.get_lf_line(z,lines)
        phi_true = np.interp(data_dict[z]['log10L'],log10L,np.log10(phi))
        err = data_dict[z]['phi_err']
        phi_obs = np.random.norma(loc=phi_true,scale=err)
        data_dict[z]['phi'] = phi_obs
    return data_dict
