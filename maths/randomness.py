import numpy as np

import maths.numpypreconditions as npp
import maths.numpyutils as npu
import utils.preconditions as pre

class NormalVariatesGenerator(object):
    def __init__(self, randomstate=None):
        self.__randomstate = npu.randomstate() if randomstate is None else randomstate
        
    def generatenormalvariates(self, cov, count):
        cov = npp.checkshapeissquare(npu.tondim2(cov))
        count = pre.checknonnegativeinteger(count)
        dim = np.shape(cov)[0]
        covroot = np.linalg.cholesky(cov)
        variates = np.reshape(self.__randomstate.normal(size=count*dim), (count, dim))
        return np.dot(variates, covroot.T)
    