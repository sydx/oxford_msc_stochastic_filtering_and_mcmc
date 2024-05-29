import numpy as np

def checksize(arg, size, message='Invalid size'):
    if np.size(arg) != size: raise ValueError(message)
    return arg

def checkshape(arg, shape, message='Invalid shape'):
    if np.shape(arg) != shape: raise ValueError(message)
    return arg

def checksameshape(arg1, arg2, message='Shapes differ'):
    s1 = np.shape(arg1)
    if s1 != np.shape(arg2): raise ValueError(message)
    return s1

def checkshapeissquare(arg, message='Shape is not square'):
    s = np.shape(arg)
    if np.size(arg) > 1:
        if len(s) != 2 or s[0] != s[1]: raise ValueError(message)
    return arg

def checkshapeisrow(arg, message='Shape is not a row'):
    s = np.shape(arg)
    if len(s) != 2 or s[0] != 1: raise ValueError(message)
    return arg
    
def checkshapeiscol(arg, message='Shape is not a column'):
    s = np.shape(arg)
    if len(s) != 2 or s[1] != 1: raise ValueError(message)
    return arg
