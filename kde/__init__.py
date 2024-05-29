import numpy as np
import numpy.linalg as la

import scipy.stats as stats

import maths.numpyutils as npu

class KernelDensityEstimator(object):
    class KernelDensityEstimate(object):
        def __init__(self, estimator, sample):
            self.__estimator = estimator
            self.__sample = sample

        def pdf(self, value):
            result = []
            for v in np.reshape(value, (np.prod(np.shape(value)[0:-1]), np.shape(value)[-1])):
                s = np.sum(self.__estimator.density(self.__estimator.bandwidthinv.dot((v - self.__sample).T).T))
                result.append(s / (self.__estimator.bandwidthdet * float(len(self.__sample))))
            return np.reshape(result, np.shape(value)[0:-1])

    def __init__(self, density, bandwidth):
        self.__density = density
        self.__bandwidth = bandwidth
        self.__bandwidthinv = la.inv(bandwidth)
        self.__bandwidthdet = la.det(bandwidth)

    def __getdensity(self):
        return self.__density

    density = property(fget=__getdensity)

    def __getbandwidth(self):
        return self.__bandwidth

    bandwidth = property(fget=__getbandwidth)

    def __getbandwidthinv(self):
        return self.__bandwidthinv

    bandwidthinv = property(fget=__getbandwidthinv)

    def __getbandwidthdet(self):
        return self.__bandwidthdet

    bandwidthdet = property(fget=__getbandwidthdet)

    def estimate(self, sample):
        return self.KernelDensityEstimate(self, sample)
