import numpy as np
import numpy.linalg as la

import scipy.integrate as integrate
import scipy.optimize as opt

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm

import maths.stats
import maths.distributions
import kde.visual


def ise(estimateddensity, theoreticaldensity):
    def integrand(*x):
        value = np.array(x)
        return (estimateddensity(value) - theoreticaldensity(value))**2

    return integrate.nquad(integrand, [[-5., 5.], [-5., 5.]])

distribution = maths.distributions.mixturedistributionE
theoreticaldensity = distribution.pdf

sample = distribution.sample(1000)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
kde.visual.make3dplot(ax, sample, theoreticaldensity)

kerneldensity = maths.distributions.MultivariateNormalDistribution([0., 0.], [[1., 0.], [0., 1.]]).pdf
bandwidth = [[1., 0.], [0., 1.]]
estimator = kde.KernelDensityEstimator(kerneldensity, bandwidth)
estimateddensity = estimator.estimate(sample).pdf

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
kde.visual.make3dplot(ax, sample, estimateddensity)

print 'Sanity checking ISE...'
print 'ISE:', ise(estimateddensity, estimateddensity)
print 'ISE:', ise(theoreticaldensity, theoreticaldensity)
print 'Computing ISE...'
print 'ISE:', ise(estimateddensity, theoreticaldensity)

kerneldensity = maths.distributions.MultivariateNormalDistribution([0., 0.], [[1., 0.], [0., 1.]]).pdf
bandwidth = [[0.31622776601, 0.], [0., 0.31622776601]]
estimator = kde.KernelDensityEstimator(kerneldensity, bandwidth)
estimateddensity = estimator.estimate(sample).pdf

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
kde.visual.make3dplot(ax, sample, estimateddensity)

print 'Computing ISE...'
print 'ISE:', ise(estimateddensity, theoreticaldensity)

sd1 = 0.21604776; sd2 = 0.25777107; cor = 0.42044755
bandwidth = maths.stats.choleskysqrt2d(sd1, sd2, cor)

estimator = kde.KernelDensityEstimator(kerneldensity, bandwidth)
estimateddensity = estimator.estimate(sample).pdf

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
kde.visual.make3dplot(ax, sample, estimateddensity)

print 'Computing (optimal) ISE...'
print 'ISE:', ise(estimateddensity, theoreticaldensity)

plt.show()
2
class BivariateGaussianKernelCostFunction(object):    
    def __init__(self, sample):
        self.__sample = sample
        n = len(sample)
        c = int(.5 * n * (n-1))
        self.__iidx = [None] * c
        self.__jidx = [None] * c
        idxidx = 0
        for i in xrange(n):
            for j in xrange(i+1, n):
                self.__iidx[idxidx] = i
                self.__jidx[idxidx] = j
                idxidx += 1
        self.__n = float(n)

    def __call__(self, params):
        print '???', params
        sd1 = params[0]
        sd2 = params[1]
        cor = params[2]

        if sd1 < 0. or sd1 > 10. or sd2 < 0. or sd2 > 10. or cor < -1. or cor > 1.:
            return np.inf

        bandwidth = maths.stats.choleskysqrt2d(sd1, sd2, cor)
        bandwidthdet = la.det(bandwidth)
        bandwidthinv = la.inv(bandwidth)

        diff = sample[self.__iidx] - sample[self.__jidx]
        temp = diff.dot(bandwidthinv.T)
        temp *= temp
        e = np.exp(np.sum(temp, axis=1))
        s = np.sum(e**(-.25) - 4 * e**(-.5))

        cost = self.__n / bandwidthdet + (2. / bandwidthdet) * s
        print '!!!', cost
        return cost / 10000.

def eigenvalueconstraint(params):
    sd1 = params[0]
    sd2 = params[1]
    cor = params[2]
    bandwidth = maths.stats.choleskysqrt2d(sd1, sd2, cor)
    bandwidthsq = bandwidth.dot(bandwidth.T)
    return -np.min(la.eigvals(bandwidthsq))

def boundconstraint(params):
    sd1 = params[0]
    sd2 = params[1]
    cor = params[2]
    if sd1 > 0. and sd1 < 10. and sd2 > 0. and sd2 < 10. and cor > -1. and cor < 1.:
        return -1.
    else:
        return 1.

costfunction = BivariateGaussianKernelCostFunction(sample)

x0 = np.array([1.2, 1.3, 0.1])
res = opt.fmin_bfgs(
    costfunction,
    x0=x0)

print res
# 0.15286931  0.13744354  0.5183952
# 0.21604776  0.25777107  0.42044755

plt.show()
