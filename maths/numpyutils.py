import numpy as np

import maths.numpypreconditions as npp

def immutablecopyof(arg):
	if isinstance(arg, np.ndarray):
		result = np.copy(arg) if arg.flags.writeable else arg
	else:
		result = np.array(arg)
		result.flags.writeable = False
	return result
		
def toscalar(arg):
	arg = npp.checksize(arg, 1)
	r = np.ndim(arg)
	if r == 1: arg = arg[0]
	elif r == 2: arg = arg[0, 0]
	return arg

def tondim1(arg):
	return np.reshape(arg, (np.size(arg),))

def tondim2(arg, ndim1tocol=False, copy=False):
	r = np.ndim(arg)
	if r == 0: arg = np.array(((arg,),))
	elif r == 1:
		arg = np.array((arg,))
		if ndim1tocol: arg = arg.T
	return np.array(arg, copy=copy)

def lowertosymmetric(a, copy=False):
	a = np.copy(a) if copy else a
	idxs = np.triu_indices_from(a)
	a[idxs] = a[(idxs[1], idxs[0])]

def uppertosymmetric(a, copy=False):
	a = np.copy(a) if copy else a
	idxs = np.triu_indices_from(a)
	a[(idxs[1], idxs[0])] = a[idxs]

def vectorised(func):
	func.__dict__['vectorised'] = True
	return func

def isvectorised(func):
	res = False
	if hasattr(func, '__call__'):
		if hasattr(func.__call__, '__dict__'):
			res |= func.__call__.__getattribute__('__dict__').get('vectorised', False)
	if not res and hasattr(func, '__dict__'):
		res = func.__getattribute__('__dict__').get('vectorised', False)
	return res

class NumericError(Exception):
	def __init__(self, message):
		super(NumericError, self).__init__(message)

__randomstate = None

def randomstate():
    global __randomstate
    if __randomstate is None:
        __randomstate = np.random.RandomState(seed=42)
    return __randomstate
