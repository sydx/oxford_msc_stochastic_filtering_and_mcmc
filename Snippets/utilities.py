import numpy as np

def toColumnVector(data):
    return np.array(data).reshape((len(data), 1))

def toMatrix(data):
    if np.ndim(data) == 1: return np.reshape(data, (1, len(data)))
    else: return np.array(data, copy=False)

def coarsen(ts, c=2):
    return ts[0::c]

