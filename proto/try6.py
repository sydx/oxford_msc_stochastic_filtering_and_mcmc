import numpy as np
import pylab

sigma = 1.
rho = .4

def compute(sigma, rho):
    means = [0., 0.]
    covs = [[1., rho*sigma], [rho*sigma, sigma*sigma]]
    vs = np.random.multivariate_normal(means, covs, 10000000)
    return np.log(np.mean(vs[:,1] * vs[:,0] * np.exp(.5 * vs[:,1])) / (rho*sigma*15/16))

def fun(sigma, rho):
    return .12643 * sigma * sigma + 0.421 * sigma - 0.07

rhos = np.linspace(-.8, .8, 5)
es = [compute(sigma, rho) for rho in rhos]
es1 = [fun(sigma, rho) for rho in rhos]

pylab.figure()
pylab.plot(rhos, es)
pylab.plot(rhos, es1)

sigmas = np.linspace(.001, 6., 50)
es = [compute(sigma, rho) for sigma in sigmas]
es1 = [fun(sigma, rho) for sigma in sigmas]

pylab.figure()
pylab.plot(sigmas, es)
pylab.plot(sigmas, es1)

print np.polyfit(sigmas, es, 2)

pylab.show()
