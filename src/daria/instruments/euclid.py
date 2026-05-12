class Euclid(object):
    def __init__(self):
        self.h = 0.6766
        self.dndV = self.__get_dndV()
        self.bias = self.__get_bias()

    def __get_dndV(self):
        return lambda z: 4e-3 * self.h**3

    def __get_bias(self):
        return lambda z: 0.7*(1+z)
    
    def get_target_prop(self):
        target_prop = {'dndV': self.dndV, 'bias': self.bias}
        return target_prop
    
    def label(self):
        return 'euclid_deep'
