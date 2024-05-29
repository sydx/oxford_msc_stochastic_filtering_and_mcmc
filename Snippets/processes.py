import numpy as np

import utilities as util

def generateBrownianMotion(ts, dim=1):
    ws = np.empty((ts.size, dim))
    ws[0] = np.zeros((1, dim))
    variates = np.random.normal(loc=0., scale=1., size=((ts.size-1, dim)))
    for i in xrange(1, ts.size):
        ws[i] = ws[i-1] + np.sqrt(ts[i]-ts[i-1])*variates[i-1]
    return ws

def generateCorrelatedBrownianMotionWithDrift(drift, volatility, ts, ws, initvalue):
    """
    >>> import simulation as sim
    >>> np.random.seed(42)
    >>> ts = sim.generateEquallySpacedTimes(1., 10)
    >>> ws = generateBrownianMotion(ts, dim=1)
    >>> generateCorrelatedBrownianMotionWithDrift([0.3, 0.4], [1.0, 3.0], ts, ws, 0.0)
    array([[ 0.        ,  0.        ],
           [ 0.19890472,  0.5411586 ],
           [ 0.18614995,  0.44733874],
           [ 0.43537946,  1.13947172],
           [ 0.97638942,  2.70694602],
           [ 0.93167162,  2.51723709],
           [ 0.8869593 ,  2.32754458],
           [ 1.44669691,  3.95120184],
           [ 1.73584182,  4.76308101],
           [ 1.61268369,  4.33805107]])

    >>> ws = generateBrownianMotion(ts, dim=2)
    >>> generateCorrelatedBrownianMotionWithDrift([0.3, 0.4], [[1.0, 0.0], [0.0, 3.0]], ts, ws, 1.0)
    array([[ 1.        ,  1.        ],
           [ 1.21418668,  0.58102675],
           [ 1.09227676,  0.86743347],
           [ 0.48785002, -0.81303992],
           [ 0.33375417, -1.7814266 ],
           [ 0.47183662, -2.64500623],
           [ 0.03440205, -1.13491301],
           [-0.00752338, -1.02294036],
           [-0.44910611, -1.52287864],
           [-0.37879858, -2.62942778]])

    >>> generateCorrelatedBrownianMotionWithDrift([0.3, 0.4], [[1.0, 0.0], [0.0, 1.0]], ts, ws, [1.0, -2.0])
    array([[ 1.        , -2.        ],
           [ 1.21418668, -2.11002812],
           [ 1.09227676, -1.98492958],
           [ 0.48785002, -2.51545775],
           [ 0.33375417, -2.80862368],
           [ 0.47183662, -3.06685393],
           [ 0.03440205, -2.53385989],
           [-0.00752338, -2.46690605],
           [-0.44910611, -2.60392251],
           [-0.37879858, -2.94314259]])
    """
    ts = util.toColumnVector(ts)
    volatility = util.toMatrix(volatility)
    return initvalue + drift*ts + np.dot(ws, volatility)

def generateGeometricBrownianMotion(pctdrift, pctvolatility, ts, ws, initvalue):
    xs = np.empty(ts.size)
    xs[0] = initvalue
    factor = pctdrift - .5*pctvolatility*pctvolatility
    for i in xrange(1, ts.size):
        xs[i] = xs[0] * np.exp(factor*ts[i] + pctvolatility*ws[i])
    return xs


if __name__ == '__main__':
    import doctest
    doctest.testmod()
