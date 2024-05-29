import numpy as np
import pylab

sigma = 1.
rho = .5

def compute(sigma, rho):
    means = [0., 0.]
    covs = [[1., rho*sigma], [rho*sigma, sigma*sigma]]
    vs = np.random.multivariate_normal(means, covs, 50000000)
    return np.log(     np.mean(   vs[:,1] * vs[:,0] * np.exp(.5 * vs[:,1])   ) / (rho * sigma)     )

def fun(sigma, rho):
    result = 0.05128323 * sigma * sigma + 1.19536232 * sigma - 1.44641948
    result = 0.09569222 * sigma * sigma + 0.44342915 * sigma
    #  0.09575427  0.47810717
    # 3/32?
    return result

#rhos = np.linspace(-.8, .8, 5)
#es = [compute(sigma, rho) for rho in rhos]
#es1 = [fun(sigma, rho) for rho in rhos]

#pylab.figure()
#pylab.plot(rhos, es)
#pylab.plot(rhos, es1)

sigmas = np.linspace(.001, 6., 50)
es = [compute(sigma, rho) for sigma in sigmas]
es1 = [fun(sigma, rho) for sigma in sigmas]

pylab.figure()
pylab.plot(sigmas, es)
pylab.plot(sigmas, es1)

coeffs = np.polyfit(sigmas, es / sigmas, 1)
print coeffs
pylab.plot(sigmas, (sigmas * sigmas) * coeffs[0] + sigmas * coeffs[1])

pylab.show()

for i in xrange(50):
    rho = -.4
    sigmas = np.linspace(.001, 6., 50)
    es = [compute(sigma, rho) for sigma in sigmas]
    print i, np.polyfit(sigmas, es / sigmas, 1)
