from hmf import MassFunction
import numpy as np

def init_hmf(Mmin=4,Mmax=16,hmf_model='Tinker10'):
    return MassFunction(z=0,Mmin=Mmin,Mmax=Mmax,dlog10m=0.01,\
                        hmf_model=hmf_model)

def get_halo_bias(hmf,z):
    hmf.update(z=z)
    g = hmf.growth_factor
    sigma = hmf._sigma_0

    # Note that this is also HMF's definition of nu
    delta_sc = 1.686
    nu = (delta_sc / sigma / g)

    y = np.log10(200)
    A = 1 + 0.24 * y * np.exp(-(4/y)**4)
    a = 0.44 * y - 0.88
    B = 0.183
    b = 1.5
    C = 0.019 + 0.107 * y + 0.19 * np.exp(-(4/y)**4)
    c = 2.4

    bias = 1 - A * (nu**a / (nu**a + delta_sc**a)) + B * nu**b + C * nu**c

    return bias
    
