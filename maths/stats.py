import numpy as np

import maths.numpyutils as npu
import utils.collections

def cor2cov(cors, vars=None, sds=None, copy=True):
    assert (vars is None and sds is not None) or (vars is not None and sds is None)
    sds = np.sqrt(vars) if vars is not None else sds
    if isinstance(cors, (utils.collections.DiagonalArray, utils.collections.SubdiagonalArray)):
        cors = cors.tonumpyarray()
    cors = npu.tondim2(cors, copy=copy)
    dim = len(vars)
    assert dim == np.shape(cors)[0] and dim == np.shape(cors)[1]
    np.fill_diagonal(cors, 1.)
    for i in range(dim):
        cors[i,:] = sds[i] * cors[i,:]
        cors[:,i] = sds[i] * cors[:,i]
    npu.lowertosymmetric(cors, copy=False)
    return cors

def choleskysqrt2d(sd1, sd2, cor):
    return np.array(((sd1, 0.), (sd2 * cor, sd2 * np.sqrt(1. - cor * cor))))

class OnlineMeanAndVarCalculator(object):
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.__n = 0
        self.__mean = 0.0
        self.__meansq = 0.0
        self.__M2 = 0.0
        
    def __get_count(self):
        return self.__n
    
    count = property(fget=__get_count)
    
    def __get_mean(self):
        return self.__mean
    
    mean = property(fget=__get_mean)
    
    def __get_meansq(self):
        return self.__meansq
    
    meansq = property(fget=__get_meansq)
    
    def __get_rms(self):
        return np.sqrt(self.meansq)
    
    rms = property(fget=__get_rms)
    
    def __get_varN(self):
        return self.__M2 / self.__n
    
    varN = property(fget=__get_varN)
    
    def __get_var(self):
        return self.__M2 / (self.__n - 1)
    
    var = property(fget=__get_var)
    
    def __get_sd(self):
        return np.sqrt(self.var)
    
    sd = property(fget=__get_sd)
    
    def __get_sdN(self):
        return np.sqrt(self.varN)
    
    sdN = property(fget=__get_sdN)
    
    def add(self, x):
        self.__n += 1
        delta = x - self.__mean
        self.__mean += delta / self.__n
        deltasq = x * x - self.__meansq
        self.__meansq += deltasq / self.__n
        if self.__n > 1:
            self.__M2 += delta * (x - self.__mean)

    def addall(self, xs):
        for x in xs: self.add(x)
        