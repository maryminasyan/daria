import numpy as np

class EllBinner(object):
    def __init__(self,ellmin=50,ellmax=1e5,nbins=30,log=True):
        self.ellmin = ellmin
        self.ellmax = ellmax
        self.nbins = nbins
        self.log = log
        self.binfunc = np.logspace if self.log else np.linspace
        
        def get_bound(extremum):
            bound = getattr(self,'ell%s' % extremum.lower())
            return np.log10(bound) if self.log else bound
        
        ellmin_use = get_bound('min')
        ellmax_use = get_bound('max')

        ell_edges = self.binfunc(ellmin_use,ellmax_use,self.nbins+1)
        self.ell_edges = ell_edges
        self.ell_bins = self.__get_ell_bins()
        self.nell = ell_edges[1:]**2 - ell_edges[:-1]**2

    def __get_ell_bins(self):
        ell_edges = self.ell_edges
        if self.log:
            return np.sqrt(ell_edges[:-1] * ell_edges[1:])
        else:
            return np.sqrt(ell_edges[:-1] + ell_edges[1:]) / 2

    def label(self):
        bintypestr = 'log' if self.log else 'lin'
        return 'ell%d-%d_%d%sbins' % (self.ellmin,self.ellmax,self.nbins,\
                                      bintypestr)
