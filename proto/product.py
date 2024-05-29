import numpy as np
import pylab
import scipy.stats as stats

sigmas = np.linspace(0.000001, 1.0, 100)
xs = np.linspace(-3., 3., 100)

for sigma in sigmas:
    ys = stats.norm.pdf(xs, loc=0., scale=sigma) * stats.norm.pdf(xs, loc=0., scale=1.)
    pylab.plot(xs, ys)

pylab.show()

