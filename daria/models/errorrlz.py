# error generator
import numpy as np

class ErrorRlz(object):
    ''' Generate some reproducible error realization '''
    def __init__(self,seed,log_err_norm=-13,err_evol=0):
        """
        Parameters
        ----------
        seed : int
            Seed for RNG
        log_err_norm : float
        err_evol : float
        """
        self.seed = seed
        self.log_err_norm = log_err_norm
        self.err_evol = err_evol

    def __get_var_Cell(self,ell,wave,z):
        err_norm = 10**self.log_err_norm
        return err_norm * (ell/1e2)**-3 * (z/1)**self.err_evol * (wave/1)**0

    def __get_std_Dell(self,ell,wave,z):
        var_Cell = self.__get_var_Cell(ell,wave,z)
        Cell_to_Dell = ell * (ell+1) / 2 / np.pi
        return Cell_to_Dell * np.sqrt(var_Cell)

    def get_error_Dell(self,ell,wave,z):
        std_Dell = self.__get_std_Dell(ell,wave,z)
        rng = np.random.default_rng(seed=self.seed)
        err_rlz = rng.normal(scale=std_Dell)
        return err_rlz

    def label(self):
        return 'err_rng%d_lognorm%d_zev%d' % (self.seed,self.log_err_norm,\
                                              self.err_evol)
