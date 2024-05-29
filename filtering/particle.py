from collections import OrderedDict
import warnings

import numpy as np
import statsmodels.api as sm

import maths.numpyutils as npu
import maths.outliers
    
class ParticleFilter(object):
    MINWEIGHTSUM = np.finfo(float).eps
    
    def __init__(self, initialdistribution, transitiondistribution, weightingfunction, particlecount, statedim=1, observationdim=1, randomstate=None, predictedobservationsampler=None, outlierthreshold=None):
        self._statedim = statedim
        self._observationdim = observationdim
        self._initialdistribution = initialdistribution
        self._transitiondistribution = transitiondistribution
        self._weightingfunction = weightingfunction
        self._particlecount = particlecount
        self._currentparticleidx = None
        self._randomstate = npu.randomstate() if randomstate is None else randomstate
        self._predictedobservationsampler = predictedobservationsampler
        
        self._priorparticles = np.empty((particlecount, statedim))
        self._resampledparticles = np.empty((particlecount, statedim))
        self._unnormalisedweights = np.empty((particlecount,))
        self._weights = np.empty((particlecount,))
        self._resampledparticlesuptodate = False
        
        self._lastobservation = None
        
        self._cachedpriormean = None
        self._cachedpriorvar = None
        self._cachedposteriormean = None
        self._cachedposteriorvar = None
        self._cachedresampledmean = None
        self._cachedresampledvar = None
        
        self.loglikelihood = 0.0        
        self.effectivesamplesize = np.NaN
        
        if self._predictedobservationsampler is not None:
            self.predictedobservationparticles = None
            self.predictedobservationkde = None
            self.predictedobservation = np.NaN
            self.innovation = np.NaN
            self.innovationvar = np.NaN
            
        assert self._predictedobservationsampler is not None or outlierthreshold is None 
        self._outlierthreshold = outlierthreshold
            
        self._context = OrderedDict()
        
        self._initialise()
        
    # An auxiliary method of the constructor. Not called anywhere else.
    def _initialise(self):
        # TODO Vectorise
        for i in range(self.particlecount):
            self._currentparticleidx = i
            self._priorparticles[i,:] = npu.tondim1(self._initialdistribution.sample())
            self._resampledparticles[i,:] = self._priorparticles[i,:]
        self._currentparticleidx = None
        self._unnormalisedweights[:] = np.NaN
        self._weights[:] = 1./self._particlecount
            
    def predict(self):
        if not self._resampledparticlesuptodate:
            self._resampledparticles[:] = self._priorparticles[:]
        if npu.isvectorised(self._transitiondistribution.sample):
            self._priorparticles = self._transitiondistribution.sample(self._resampledparticles, stochfilter=self)
        else:
            for i in range(self.particlecount):
                self._currentparticleidx = i
                self._priorparticles[i,:] = npu.tondim1(self._transitiondistribution.sample(self._resampledparticles[i,:], stochfilter=self))
            self._currentparticleidx = None

        self._resampledparticlesuptodate = False
        self._cachedpriormean = None
        self._cachedpriorvar = None
        
        # TODO Vectorise
        # TODO using fft kde - assumes all weights are equal!
        # TODO This only works when statedim == 1
        if self._predictedobservationsampler is not None:
            if npu.isvectorised(self._predictedobservationsampler):
                self.predictedobservationparticles = self._predictedobservationsampler(self._priorparticles, self)
            else:
                self.predictedobservationparticles = np.empty((self.particlecount, self._observationdim))
                for i in range(self.particlecount):
                    self._currentparticleidx = i
                    self.predictedobservationparticles[i,:] = self._predictedobservationsampler(self._priorparticles[i,:], self)
                self._currentparticleidx = None
            self.predictedobservation = np.average(self.predictedobservationparticles, weights=self._weights, axis=0)
            self.predictedobservationkde = sm.nonparametric.KDEUnivariate(self.predictedobservationparticles)
            #fft=False, weights=self._weights
            self.predictedobservationkde.fit()
            # import matplotlib.pyplot as plt
            # fig = plt.figure()
            #x_grid = np.linspace(-4.5, 3.5, 1000)
            #plt.plot(x_grid, kde.evaluate(x_grid))
            #plt.show()
            self.innovationvar = np.var(self.predictedobservationparticles) + self.predictedobservationkde.bw * self.predictedobservationkde.bw
            
    def _weight(self, observation):
        if self._predictedobservationsampler is not None:
            self.innovation = observation - self.predictedobservation
        
        weightsum = 0.

        if npu.isvectorised(self._weightingfunction):
            self._unnormalisedweights = npu.tondim1(self._weightingfunction(observation, self._priorparticles, self))
            weightsum = np.sum(self._unnormalisedweights)
        else:
            for i in range(self.particlecount):
                self._currentparticleidx = i
                self._unnormalisedweights[i] = npu.toscalar(self._weightingfunction(observation, self._priorparticles[i,:], self))
                weightsum += self._unnormalisedweights[i]
            self._currentparticleidx = None
                
        if weightsum < ParticleFilter.MINWEIGHTSUM:
            warnings.warn('The sum of weights is less than MINWEIGHTSUM')
            #self._unnormalisedweights[:] = 1. / self.particlecount
            #weightsum = 1.
        
        self._weights = self._unnormalisedweights / weightsum
        
        self.effectivesamplesize = 1. / np.sum(np.square(self._weights))

        self.loglikelihood += np.log(np.sum(self._unnormalisedweights) / self.particlecount)
        
        self._lastobservation = observation
        
        self._cachedposteriormean = None
        self._cachedposteriorvar = None
        
    def _resample(self):
        raise NotImplementedError('Pure virtual method')
        
    def observe(self, observation):
        if self._outlierthreshold is not None:
            if maths.outliers.isoutlier(self.predictedobservationparticles, self.predictedobservationkde.bw, observation, self._outlierthreshold, 100000, self._randomstate):
                print('OUTLIER!!!')
                self._resampledparticles = np.copy(self._priorparticles)
                return False
            else:
                # print('NOT AN OUTLIER!!!')
                pass
        self._weight(observation)
        self._resample()
        return True
        
    def _getpriorparticles(self):
        return npu.immutablecopyof(self._priorparticles)
    
    priorparticles = property(fget=_getpriorparticles)
    
    def _getresampledparticles(self):
        return npu.immutablecopyof(self._resampledparticles)
    
    resampledparticles = property(fget=_getresampledparticles)
    
    def _getunnormalisedweights(self):
        return npu.immutablecopyof(self._unnormalisedweights)
    
    unnormalisedweights = property(fget=_getunnormalisedweights)
    
    def _getweights(self):
        return npu.immutablecopyof(self._weights)
    
    weights = property(fget=_getweights)
    
    def priormean(self):
        if self._cachedpriormean is None:
            self._cachedpriormean = np.average(self._priorparticles, axis=0)
        return self._cachedpriormean
    
    def priorvar(self):
        if self._cachedpriorvar is None:
            self._cachedpriorvar = np.average((self._priorparticles - self.priormean())**2, axis=0)
        return self._cachedpriorvar
    
    def posteriormean(self):
        if self._cachedposteriormean is None:
            self._cachedposteriormean = np.average(self._priorparticles, weights=self._weights, axis=0)
        return self._cachedposteriormean

    def posteriorvar(self):
        if self._cachedposteriorvar is None:
            self._cachedposteriorvar = np.average((self._priorparticles - self.posteriormean())**2, weights=self._weights, axis=0) 
        return self._cachedposteriorvar

    def resampledmean(self):
        if self._cachedresampledmean is None:
            self._cachedresampledmean = np.average(self._resampledparticles, axis=0)
        return self._cachedresampledmean
    
    def resampledvar(self):
        if self._cachedresampledvar is None:
            self._cachedresampledvar = np.average((self._resampledparticles - self.resampledmean())**2, axis=0)
        return self._cachedresampledvar
    
    @property
    def mean(self): return self.resampledmean()
    
    @property
    def var(self): return self.resampledvar()
    
    @property
    def lastobservation(self): return self._lastobservation

    @property
    def particlecount(self): return self._particlecount
    
    @property
    def currentparticleidx(self): return self._currentparticleidx
    
    @property
    def context(self): return self._context
    
class MultinomialResamplingParticleFilter(ParticleFilter):
    def _resample(self):
        counts = self._randomstate.multinomial(self.particlecount, self._weights)
        particleidx = 0
        for i in range(self.particlecount):
            for j in range(counts[i]):  # @UnusedVariable
                self._resampledparticles[particleidx,:] = self._priorparticles[i,:]
                particleidx += 1
        
        self._resampledparticlesuptodate = True
        self._cachedresampledmean = None
        self._cachedresampledvar = None

class RegularisedResamplingParticleFilter(ParticleFilter):
    def _resample(self):
        # TODO This only works when statedim == 1
        # TODO Vectorise
        kde = sm.nonparametric.KDEUnivariate(self._priorparticles)
        kde.fit(fft=False, weights=self._weights)
        counts = self._randomstate.multinomial(self.particlecount, self._weights)
        particleidx = 0
        bwfactor = .5
        for i in range(self.particlecount):
            for j in range(counts[i]):  # @UnusedVariable
                self._resampledparticles[particleidx,:] = self._priorparticles[i,:]
                particleidx += 1
        self._resampledparticles[:] += bwfactor * kde.bw * self._randomstate.normal(size=(self.particlecount, 1))
        
        self._resampledparticlesuptodate = True
        self._cachedresampledmean = None
        self._cachedresampledvar = None
            
class SmoothResamplingParticleFilter(ParticleFilter):
    def _resample(self):
        newweights = np.empty((self.particlecount+1,))
        newweights[0] = .5*self._weights[0]
        newweights[self.particlecount] = .5*self._weights[self.particlecount-1]
        for i in range(1, self.particlecount):
            newweights[i] = .5*(self._weights[i] + self._weights[i-1])

        uniforms = self._randomstate.uniform(size=self.particlecount)
        uniforms.sort()
        newuniforms = np.empty(shape=(self.particlecount,))
        regions = np.empty(shape=(self.particlecount,), dtype=np.int)
        s = 0
        j = 0
        for i in range(self.particlecount + 1):
            s = s + newweights[i]
            while j < self.particlecount and uniforms[j] <= s:
                regions[j] = i
                newuniforms[j] = (uniforms[j] - (s - newweights[i])) / newweights[i]
                j += 1
                
        for i in range(self.particlecount):
            if regions[i] == 0:
                self._resampledparticles[i,:] = self._priorparticles[0,:]
            if regions[i] == self.particlecount:
                self._resampledparticles[i,:] = self._priorparticles[self.particlecount-1,:]
            else:
                self._resampledparticles[i,:] = (self._priorparticles[regions[i],:] - self._priorparticles[regions[i]-1,:]) * newuniforms[i] + self._priorparticles[regions[i]-1,:]   
            
        self._resampledparticlesuptodate = True
        self._cachedresampledmean = None
        self._cachedresampledvar = None
