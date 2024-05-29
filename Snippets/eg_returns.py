import numpy as np
import pylab

simplenetreturns = np.linspace(-1., 1., 100)
logreturns = np.log(simplenetreturns + 1.)

pylab.plot(simplenetreturns, simplenetreturns, label='simple net return')
pylab.plot(simplenetreturns, logreturns, label='logarithmic return')
pylab.axhline(0., color='black')
pylab.axvline(0., color='black')
pylab.xlabel('simple net return')
pylab.legend(loc=4)

pylab.figure()
pylab.plot(simplenetreturns, simplenetreturns - logreturns, label='simple net return - logarithmic return')
pylab.xlim((-0.5, 0.5))
pylab.ylim((0., 0.2))
pylab.xlabel('simple net return')
pylab.legend(loc=1)

pylab.show()
