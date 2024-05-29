import numpy as np

def eulerMaruyama(a, b, x0, T, K, vardim=1):
    deltat = T / K
    # TODO: Scale the variates by multiplication
    # NB: scale below is standard deviation, not variance!
    variates = np.random.normal(loc=0.0, scale=np.sqrt(deltat), size=(vardim, K))
    X = np.zeros((np.size(x0), K+1))
    X[:,[0]] = x0
    for k in xrange(1, K+1):
        prevt = 0.0 + (k - 1) * deltat
        prevX = X[:,[k-1]]
        X[:,[k]] = prevX + \
            a(prevt, prevX) * deltat + \
            np.dot(b(prevt, prevX), variates[:,[k-1]])
    return X
