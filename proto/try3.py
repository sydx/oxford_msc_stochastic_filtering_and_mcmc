import numpy as np
import pylab

sigma = 2.
rho = .4

def compute(sigma, rho):
    means = [0., 0.]
    covs = [[1., rho*sigma], [rho*sigma, sigma*sigma]]
    vs = np.random.multivariate_normal(means, covs, 2000000)
    return np.mean(vs[:,1] * vs[:,0] * np.exp(.5 * vs[:,1]))

def fun(sigma, rho):
    return .5 * rho * sigma * np.exp(.2 * sigma * sigma + .2 * sigma)

rhos = np.linspace(-.8, .8, 10)
es = [compute(sigma, rho) for rho in rhos]
es1 = [fun(sigma, rho) for rho in rhos]

pylab.figure()
pylab.plot(rhos, es)
pylab.plot(rhos, es1)

sigmas = np.linspace(.1, 5., 20)
es = [compute(sigma, rho) for sigma in sigmas]
es1 = [fun(sigma, rho) for sigma in sigmas]

pylab.figure()
pylab.plot(sigmas, es)
pylab.plot(sigmas, es1)

pylab.show()

