import numpy as np
import pylab
from euler_maruyama import eulerMaruyama

T = 4.0
K = 200
vardim = 3

a = lambda  t, X : np.array([[0.1, 0.2, 0.2]]).T
b = lambda t, X : np.array([
    [100.0, -50.0, 0.0],
    [-50.0, 100.0, 0.0],
    [0.0, 0.0, 100.0]])

x0 = np.array([[-1.5, 2.3, 3.5]]).T

times = np.linspace(0.0, T, K+1)

X = eulerMaruyama(a, b, x0, T, K, vardim)

pylab.plot(times, X.T)
pylab.show()
