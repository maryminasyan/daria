import numpy as np
from ..utils.line_info import line_rwaves, line_names

def get_streak_mtx(line,z_edges_2d,wl_centers,wl_edges):
    line_rwave = line_rwaves[line]
    dz = np.diff(z_edges_2d,axis=1)[:,0]
    z_e = z_edges_2d[:,0]
    mtx = np.zeros((len(wl_centers),len(z_edges_2d)))
    for i, ch in enumerate(wl_centers):
        zlo,zhi = (wl_edges[i] * 1e4 / line_rwave) - 1
        if (zlo > z_edges_2d[-1,-1]):
            continue
        if (zhi < z_edges_2d[0,0]):
            continue
        jlo = np.digitize(zlo,bins=z_e)
        jhi = np.digitize(zhi,bins=z_e)
        if jlo > 0:
            jlo -= 1
        if jhi > 0:
            jhi -= 1

        if jlo == jhi:
            mtx[i,jlo] = (zhi - zlo) / dz[jlo]
            continue

        mtx[i,jlo] = (z_edges_2d[jlo,1] - zlo) / dz[jlo]
        mtx[i,jhi] = (zhi - z_edges_2d[jhi,0]) / dz[jhi]

        if jhi > (jlo + 1):
            for j in range(jlo+1,jhi):
                mtx[i,j] = 1

    return mtx

def fit_z_label(fit_z):
    if isinstance(fit_z,str):
        zlabel = 'all'
    else:
        zlabel = '_'.join(f'{z}:.2f' for z in fit_z)
    return 'z' + zlabel

def fit_lines_label(fit_lines):
    if fit_lines == 'all':
        lines_label = '_all'
    else:
        lines_label = '_'.join(fit_lines)
    return 'lines' + lines_label

def get_lines_fit(fit_lines):
    if fit_lines == 'all':
        return line_names
    else:
        return fit_lines
