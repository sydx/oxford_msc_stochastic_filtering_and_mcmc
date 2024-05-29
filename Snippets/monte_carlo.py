import numpy as np

def eulerMaruyama(drift, volatility, ts, ws, initvalue):
    xs = np.empty(ts.size)
    xs[0] = initvalue
    for i in xrange(1, ts.size):
        deltat = ts[i] - ts[i-1]
        deltaw = ws[i] - ws[i-1]
        av = drift(xs[i-1])
        bv = volatility(xs[i-1])
        xs[i] = xs[i-1] + av*deltat + bv*deltaw
    return xs

def milstein(drift, volatility, ts, ws, initvalue):
    xs = np.empty(ts.size)
    xs[0] = initvalue
    for i in xrange(1, ts.size):
        deltat = ts[i] - ts[i-1]
        deltaw = ws[i] - ws[i-1]
        av = drift(xs[i-1])
        bv = volatility(xs[i-1])
        xs[i] = xs[i-1] + av*deltat + bv*deltaw + .5*bv*bv*(deltaw*deltaw-deltat)
    return xs
