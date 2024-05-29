import numpy as np
import matplotlib.pyplot as plt

import filtering.generation as gen
import filtering.kalman
import filtering.run
import filtering.visualisation
import maths.numpyutils as npu

OBSERVATIONNOISEFACTOR = .5

class ARMAGenerator(object):
    def __init__(self, timecount, ar, ma, var=1., const=0., randomstate=None):
        self.__timecount = timecount
        self.__ar = npu.tondim1(npu.immutablecopyof(ar))
        self.__ma = npu.tondim1(npu.immutablecopyof(ma))
        self.__var = var
        self.__const = const
        self.__randomstate = npu.randomstate() if randomstate is None else randomstate
        
    def generate(self):
        generator = gen.StateSpaceModelDataGenerator(self.__randomstate)
        generator.addnoise('statenoise', var=self.__var, lag=0)
        generator.addnoise('observationnoise', var=self.__var * OBSERVATIONNOISEFACTOR, lag=0)
        def state(time, processname, data):
            state = self.__const
            for i in range(len(self.__ar)):
                if time - i - 1 >= 0: state += self.__ar[i] * data.process('state', time - i - 1)
            for i in range(len(self.__ma)):
                if time - i - 1 >= 0: state += self.__ma[i] * data.noise('statenoise', time - i - 1)
            return state
        generator.addprocess('state', state)
        def observation(time, processname, data):
            return data.process('state', time) + data.noise('observationnoise', time)
        generator.addprocess('observation', observation)
        generator.settimecount(self.__timecount)
        data = generator.generate()
        return data.processdf()

ar = [.75, -.4]
ma = [.7]
p = len(ar)
q = len(ma)
const = 0.
var = .1

generator = ARMAGenerator(timecount=100, ar=ar, ma=ma, const=const, var=var)

df = generator.generate()
plt.plot(df['state'], 'r', label='true state')
plt.plot(df['observation'], 'gx', label='noisy observation')
plt.legend(loc='upper right', fancybox=True, framealpha=0.5, prop={'size': 8})
plt.xlabel('time')

m = max(p, q+1)
ar = np.array([ar[i] if i < p else 0. for i in range(m)])
ma = np.array([ma[i] if i < q else 0. for i in range(m)])
F = np.zeros((m, m))
F[:,0] = ar
F[0:m-1,1:] = np.eye(m-1)
Wdiag = np.empty((m, 1))
Wdiag[0,0] = 1.
Wdiag[1:,:] = ma[:-1]
W = np.zeros((m, m))
for i in range(m):
    W[i,i] = Wdiag[i,0]
H = np.zeros((1, m))
H[0,0] = 1.

x0 = np.zeros((m, 1))
P0 = np.eye(m) * OBSERVATIONNOISEFACTOR * var
Q = np.eye(m) * var
R = OBSERVATIONNOISEFACTOR * var
a = np.zeros((m, 1))
b = np.zeros((1, 1))
V = np.eye(1)
stochfilter = filtering.kalman.KalmanFilter(x0, P0, Q, R, F, H, a, b, W, V)
filterrundata = filtering.run.runfilter(df, None, stochfilter, 'state', 'observation')

fig = plt.figure()
filtering.visualisation.makefilterrunplot(fig, filterrundata.filterrundf)        

plt.show()
