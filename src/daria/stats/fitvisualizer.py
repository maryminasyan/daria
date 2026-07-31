import numpy as np
import matplotlib.pyplot as plt
import functools

def plot_style(method):
    @functools.wraps(method)
    def wrapper(self,*args,**kwargs):
        with plt.style.context(self.style):
            return method(self,*args,**kwargs)
    return wrapper

class FitVisualizer(object):
    ''' Convenience tool to quickly plot fit results. '''
    def __init__(self,fit_results,model=None,style='default'):
        '''
        fit_results: Dict of emcee fit results. Must have the following items:
            - chain, with shape (nwalkers, nsteps, nparams)
            - fchain, with shape (nwalkers*nsteps, nparams)
            - params, i.e., list of parameter names
            - lnprob, with shape (nwalkers, nsteps)
            - data (i.e., the data fed into the emcee sampler object)
        model: Model used to run fit (optional unless you want to use the
            SFRD plotting convenience method)
        style: default (light background) or dark_background
        '''
        nwalkers, nsteps, nparams = fit_results['chain'].shape
        self.nwalkers = nwalkers
        self.nsteps = nsteps
        self.nparams = nparams
        self.params = fit_results['params']
        self.lnprob = fit_results['lnprob']
        self.chain = fit_results['chain']
        self.fchain = fit_results['fchain']
        self.data = fit_results['data']
        self.style = style.lower() # default or dark_background
        self.model = model

    def get_thin_chain(self,discard=1000,thin=200):
        return self.chain[:,discard::thin,:]
    
    def get_thin_fchain(self,discard=1000,thin=200):
        thin_chain = self.get_thin_chain(discard=discard,thin=thin)
        return thin_chain.reshape(-1,thin_chain.shape[-1])

    def make_param_dict(self,params_list):
        param_dict = {}
        for i, p in enumerate(params_list):
            param_dict[self.params[i]] = p
        return param_dict
            
    @plot_style
    def plot_lnprob_v_step(self,alpha=0.01,dpi=200,layout='tight',\
                           fn=None,xlim=None,ylim=None):
        fig, ax = plt.subplots(dpi=dpi,layout=layout)

        for i in range(self.nwalkers):
            ax.semilogy(np.abs(self.lnprob[i,:]),alpha=alpha,color='tab:blue')

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
        ax.set_xlabel('Step number')
        ax.set_ylabel(r'$\log |\mathcal{L}|$')
            
        if fn is not None:
            fig.savefig(fn)

        return fig, ax

    @plot_style
    def plot_params_v_step(self,alpha=0.01,dpi=200,layout='tight',fn=None,\
                           xlim=None,ylim=None,true_vals=None):
        nparams = self.nparams
        fig, axes = plt.subplots(nparams,figsize=(6,4*nparams),dpi=dpi,\
                                 layout=layout)

        for j in range(nparams):
            for i in range(self.nwalkers):
                axes[j].plot(self.chain[i,:,j],alpha=0.05,color='tab:blue')
            if true_vals is not None:
                axes[j].hlines(true_vals[j],xmin=0,xmax=shape,color='r')
            axes[j].set_ylabel(self.params[j])

        axes[-1].set_xlabel('Step number')

        if fn is not None:
            fig.savefig(fn)

        return fig, axes
    
    @plot_style
    def plot_sfrd(self,z,plot_type='models',ref_pts=None,ref_sim=None,
                  true_vals=None,sfr_lim=(0.3,np.inf),label='Fit',\
                  dpi=200,layout='tight',fn=None,xlim=None,ylim=None,\
                  return_sfrds=False,grid=True,discard=1000,thin=200):
        '''
        Plot SFRD vs. z fit results, either in form of a subset of models,
        or 1 and/or 2 sigma contours. Can also overlay the true input SFRD
        (if fitting to synthetic data), or reference data points/curves from
        the literature, although you will have to provide all those things
        yourself.
        
        plot_type:
           - 'models': Show some subset of individual models
           - '1sig': 16/84 percentile as shaded region
           - '2sig' 5/95 percentile as shaded region
           - '1-2sig': 16/84 percentile as darker shaded region and 5/95 as
                 lighter shaded region
        ref_pts:
           - Overlay observational reference points, e.g., from HiZELS. For now
                 the data must be provided by the user.
        ref_sim:
           - Overlay reference intervals from, e.g., MD14. For now the data
                 must be provided by the user.
        true_vals:
          - Overlay points/line corresponding to the true input model.
        return_sfrD:
          - Return computed SFRDs from all the models defined by the thinned
            fchain parameters, defaults to False.
        '''
        fig, ax = plt.subplots(dpi=dpi,layout=layout)
        fchain = self.get_thin_fchain(discard=discard,thin=thin)
        sfrds_arr = np.zeros((fchain.shape[0],len(z)))
        model = self.model
        for i in range(fchain.shape[0]):
            param_dict = self.make_param_dict(fchain[-i])
            model.update(**param_dict)
            sfrd = model.get_sfrd(z,sfr_lim=sfr_lim)
            sfrds_arr[i][:] = sfrd

        if plot_type == 'models':
            for i in range(1000):
                ax.plot(z,sfrds_arr[i],alpha=0.01,color='tab:blue')
        else:
            per_16 = np.percentile(sfrds_arr,16,axis=0)
            per_84 = np.percentile(sfrds_arr,84,axis=0)
            per_5 = np.percentile(sfrds_arr,5,axis=0)
            per_95 = np.percentile(sfrds_arr,95,axis=0)
            if plot_type == '1sig':
                ax.fill_between(z,per_16,y2=per_84,alpha=0.3,\
                                label=label)
            elif plot_type == '2sig':
                ax.fill_between(z,per_5,y2=per_95,alpha=0.7,\
                                label=label)
            elif plot_type == '1-2sig':
                ax.fill_between(z,per_16,y2=per_84,alpha=0.7)
                ax.fill_between(z,per_5,y2=per_95,alpha=0.3)

        if ref_pts is not None:
            ax.errorbar(z,ref_pts[0],yerr=ref_pts[1],label='Data',
                        fmt='.',color='tab:pink')
        if ref_sim is not None:
            ax.plot(z,ref_sim,label='Sim',ls='--')
        if true_vals is not None:
            ax.plot(z,true_vals,label='Truth',color='r')

        ax.set_xlabel('z',size='x-large')
        ax.set_ylabel(r'$\rho_{\rm{SFR}} \ [M_{\odot} \ \rm{yr}^{-1} \ \rm{cMpc}^{-3}]$',size='x-large')
        if grid:
            ax.grid(alpha=0.2)
        ax.semilogy()
        ax.tick_params(labelsize=12)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.legend(fontsize='large')

        if fn is not None:
            fig.savefig(fn)

        if return_sfrds:
            return fig, ax, sfrds_arr
        
        return fig, ax
