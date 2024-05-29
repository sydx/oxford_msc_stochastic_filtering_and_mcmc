from collections import OrderedDict
import copy

import numpy as np

import maths.numpyutils as npu
import maths.randomness as rnd
import maths.stats
import utils.collections

# If we have noises u, v, w with lags 1, 0, 3, respecitvely, then the
# correlations apply to u(t+1), v(t), and w(t+3).
class StateSpaceModelDataGenerator(object):
    class Data(object):
        def __init__(self):
            self._timecount = None
            
            self._noisenames = OrderedDict()
            self._jumpnames = OrderedDict()
            self._processnames = OrderedDict()
            
            self._noises = None
            self._jumpflags = None
            self._jumps = None
            self._processes = None
        
        def noise(self, name, time):
            return self._noises[time, self._noisenames[name]]
        
        def isjump(self, name, time):
            return self._jumpflags[time, self._jumpnames[name]]
        
        def jump(self, name, time):
            return self._jumps[time, self._jumpnames[name]]
        
        def process(self, name, time):
            return self._processes[time, self._processnames[name]]
        
        def copy(self):
            data = StateSpaceModelDataGenerator.Data()
            data._timecount = self._timecount
            data._noisenames = copy.copy(self._noisenames)
            data._jumpnames = copy.copy(self._jumpnames)
            data._processnames = copy.copy(self._processnames)
            data._noises = np.copy(self._noises)
            data._jumpflags = np.copy(self._jumpflags)
            data._jumps = np.copy(self._jumps)
            data._processes = np.copy(self._processes)
            return data
        
        def times(self):
            return range(self._timecount)
        
        def noisedf(self, colname=lambda idx, name: name):
            from pandas import DataFrame
            series = OrderedDict()
            for idx, name in enumerate(self._noisenames):
                series[colname(idx, name)] = self._noises[:,idx]
            return DataFrame(series, index=self.times())
        
        def jumpdf(self, colname=lambda idx, name: name):
            from pandas import DataFrame
            series = OrderedDict()
            for idx, name in enumerate(self._jumpnames):
                series[colname(idx, name)] = self._jumps[:,idx]
            return DataFrame(series, index=self.times())
        
        def jumpflagdf(self, colname=lambda idx, name: name):
            from pandas import DataFrame
            series = OrderedDict()
            for idx, name in enumerate(self._jumpnames):
                series[colname(idx, name)] = self._jumpflags[:,idx]
            return DataFrame(series, index=self.times())
        
        def processdf(self, colname=lambda idx, name: name):
            from pandas import DataFrame
            series = OrderedDict()
            for idx, name in enumerate(self._processnames):
                series[colname(idx, name)] = self._processes[:,idx]        
            return DataFrame(series, index=self.times())
            
    def __init__(self, randomstate=None):
        self.__randomstate = npu.randomstate() if randomstate is None else randomstate
        
        self.__noisevars = []
        self.__noiselags = []
        self.__noisecors = None
        
        self.__jumpintensities = []
        self.__jumpsds = []
        
        self.__processfuncs = []
        
        self.__data = StateSpaceModelDataGenerator.Data()
                
    def settimecount(self, timecount):
        self.__data._timecount = timecount
    
    def addnoise(self, name, var=None, sd=None, lag=0):
        assert (var is None and sd is not None) or (var is not None and sd is None)
        if var is None: var = sd*sd
        assert name not in self.__data._noisenames
        idx = len(self.__noisevars)
        self.__data._noisenames[name] = idx
        self.__noisevars.append(var)
        self.__noiselags.append(lag)
        return idx
    
    def addjump(self, name, intensity, var=None, sd=None):
        assert 0. <= intensity and intensity <= 1.
        assert (var is None and sd is not None) or (var is not None and sd is None)
        if sd is None: sd = np.sqrt(var)
        assert name not in self.__data._jumpnames
        idx = len(self.__jumpsds)
        self.__data._jumpnames[name] = idx
        self.__jumpintensities.append(intensity)
        self.__jumpsds.append(sd)
        return idx
    
    def addprocess(self, name, func):
        assert name not in self.__data._processnames
        idx = len(self.__processfuncs)
        self.__data._processnames[name] = idx
        self.__processfuncs.append(func)
        return idx
        
    def setnoisecors(self, cors):
        cors = utils.collections.SubdiagonalArray.create(cors)
        assert cors.dim == len(self.__noisevars)
        self.__noisecors = cors

    def __generatenoises(self):        
        maxlag = max(self.__noiselags)
        variatecount = self.__data._timecount + maxlag
        
        covmatrix = maths.stats.cor2cov(self.__noisecors, vars=self.__noisevars)
        variategenerator = rnd.NormalVariatesGenerator(randomstate=self.__randomstate)
        self.__data._noises = variategenerator.generatenormalvariates(covmatrix, variatecount)
        for i, lag in enumerate(self.__noiselags):
            self.__data._noises[:,i] = np.roll(self.__data._noises[:,i], lag)
        self.__data._noises = self.__data._noises[maxlag:,:]
        
    def __generatejumps(self):
        jumpcount = len(self.__jumpsds)
        intensityvariates = self.__randomstate.uniform(size=(self.__data._timecount, jumpcount))
        self.__data._jumpflags = intensityvariates < self.__jumpintensities
        jumpvariates = self.__randomstate.normal(size=np.count_nonzero(self.__data._jumpflags))
        self.__data._jumps = np.zeros((self.__data._timecount, jumpcount))
        self.__data._jumps[np.where(self.__data._jumpflags)] = jumpvariates
        self.__data._jumps *= self.__jumpsds
        
    def __validate(self):
        assert self.__data._timecount is not None
        noisecount = len(self.__data._noisenames)
        assert len(self.__noisevars) == noisecount
        assert len(self.__noiselags) == noisecount
        assert 0 in self.__noiselags
        if self.__noisecors is None:
            self.__noisecors = np.eye(noisecount)
        jumpcount = len(self.__data._jumpnames)
        assert len(self.__jumpintensities) == jumpcount
        assert len(self.__jumpsds) == jumpcount
        processcount = len(self.__data._processnames)
        assert len(self.__processfuncs) == processcount
        
    def generate(self):
        self.__validate()
        self.__generatenoises()
        self.__generatejumps()
        processcount = len(self.__data._processnames)
        self.__data._processes = np.empty((self.__data._timecount, processcount))
        self.__data._processes[:] = np.NAN
        for time in range(self.__data._timecount):
            for pi, (pn, pf) in enumerate(zip(self.__data._processnames, self.__processfuncs)):
                self.__data._processes[time, pi] = pf(time, pn, self.__data)
        return self.__data.copy()
