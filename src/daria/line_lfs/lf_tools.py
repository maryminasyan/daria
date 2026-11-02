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
        phi_obs = np.random.normal(loc=phi_true,scale=err)
        data_dict[z]['phi'] = phi_obs
    return data_dict

def get_lf_xlabel(line):
    from ..utils.line_info import line_labels
    line_label = line_labels[line]
    xlabel = rf'$\log_{{10}} L_{{\mathrm{{{line_label}}}}} / [\rm{{erg}} \ \rm{{s}}^{{-1}}]$'
    return xlabel

def plot_lf_compilation(data,line,vrange,with_center=False,dpi=150,\
                        xlim=None,ylim=None,cmap='rainbow',alpha_grid=0.2,
                        alpha_lf=0.8,label_fontsize='x-large',ticksize=12):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, LogNorm
    from matplotlib.cm import ScalarMappable

    z_line = np.array(list(data.keys()))
    z_line.sort()
    cmap = plt.get_cmap(cmap)
    vmin, vmax = vrange
    norm = Normalize(vmin=vmin,vmax=vmax)

    fig, ax = plt.subplots(dpi=dpi)
    if alpha_grid:
        ax.grid(alpha=alpha_grid)

    for z in z_line:
        log10L = data[z]['log10L']
        phi = data[z]['phi']
        phi_err = data[z]['phi_err']

        color = cmap(norm(z))

        if with_center:
            ax.plot(log10L,phi,color=color,ls='-',lw=1,alpha=alpha_lf)
        ax.fill_between(log10L,phi+phi_err,y2=phi-phi_err,\
                        color=color,alpha=alpha_lf)

    if xlim is not None:
        ax.set_xlim(xlim[0],xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0],ylim[1])

    xlabel = get_lf_xlabel(line)
    ax.set_xlabel(xlabel,fontsize=label_fontsize)
    ax.set_ylabel(r'$\phi(L) \ [\rm{cMpc}^{-3} \ \rm{dex}^{-1}]$',\
                  fontsize=label_fontsize)
    ax.tick_params(axis='both',which='major',labelsize=ticksize)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(r'$\rm{z}$',fontsize=label_fontsize)
    cbar.ax.tick_params(labelsize=label_fontsize)
    
    return fig, ax
