import numpy as np

class LinearDataBinner(object):
    def __init__(self,datamin,datamax,delta):
        self.datamin = datamin
        self.datamax = datamax
        self.delta = delta

    def centers(self):
        return np.arange(self.datamin,self.datamax,self.delta)

    def edges(self,dim=1):
        ''' edges of bins in 1- or 2-dimensional form '''
        centers = self.centers()
        if dim == 1:
            delta = self.delta
            return np.concatenate(([centers[0] - 0.5 * delta],\
                                    centers + 0.5 * delta))
        else:
            assert dim == 2, '`dim` can only be 1 or 2'
            edges = self.edges()
            return np.vstack((edges[0:-1],edges[1:])).T

    def label(self,data=''):
        return '%s%s-%s_delta%s' % (data,self.datamin,self.datamax,self.delta)
