import math

import numpy as np
import scipy.stats

import maths.numpyutils as npu
from maths.numpyutils import vectorised

class SVLJLogVarTransitionDistribution(object):
    def __init__(self, params, randomstate=None):
        self.__params = params
        self.__randomstate = npu.randomstate() if randomstate is None else randomstate
        
        self.__oneminuspersistence = 1. - params.persistence
        self.__jumpvar = params.jumpvol * params.jumpvol
        self.__oneminusjumpintensity = 1. - params.jumpintensity
    
    @vectorised    
    def __jumpreturnshockmean(self, expstate, observation):
        numerator = observation * np.sqrt(expstate)
        denominator = expstate + self.__jumpvar
        return numerator / denominator
    
    @vectorised
    def __jumpreturnshockvol(self, expstate):
        return self.__params.jumpvol / np.sqrt(expstate + self.__jumpvar)
    
    @vectorised
    def __nojumpreturnshock(self, expstate, observation):
        return observation / np.sqrt(expstate)
    
    @vectorised
    def __condjumpprobability(self, expstate, observation):
        # Note that scipy.stats.norm.pdf takes the standard deviation (rather
        # than variance) as one of its arguments
        if self.__params.jumpintensity < np.finfo(float).eps:
            return np.zeros(np.shape(expstate))
        if self.__oneminusjumpintensity < np.finfo(float).eps:
            return np.ones(np.shape(expstate))
        numerator = scipy.stats.norm.pdf(observation, 0., np.sqrt(expstate + self.__jumpvar)) * self.__params.jumpintensity
        denominator = numerator + scipy.stats.norm.pdf(observation, 0., np.sqrt(expstate)) * self.__oneminusjumpintensity
        return numerator / denominator
    
    @vectorised
    def __uniformvariatethreshold(self, nojumpreturnshock, jumpreturnshockmean, jumpreturnshockvol, condjumpprobability):
        cdfarg = (nojumpreturnshock - jumpreturnshockmean) / jumpreturnshockvol
        return scipy.stats.norm.cdf(cdfarg) * condjumpprobability
        
    @vectorised
    def __samplereturnshock(self, expstate, observation):
        nojumpreturnshock = self.__nojumpreturnshock(expstate, observation)
        jumpreturnshockmean = self.__jumpreturnshockmean(expstate, observation)
        jumpreturnshockvol = self.__jumpreturnshockvol(expstate)
        condjumpprobability = self.__condjumpprobability(expstate, observation)
        oneminuscondjumpprobability = 1. - condjumpprobability
        
        threshold = self.__uniformvariatethreshold(nojumpreturnshock, jumpreturnshockmean, jumpreturnshockvol, condjumpprobability)
        
        u = self.__randomstate.uniform(size=np.shape(expstate))
        
        # These are NumPy boolean indices
        case1 = u <= threshold
        case3 = u > threshold + oneminuscondjumpprobability
        case2 = ~(case1 + case3)
        
        returnshock = np.empty(shape=np.shape(expstate))
        
        # PPF is percent point function, i.e. inverse CDF
        ppfarg = u[case1] / condjumpprobability[case1]
        returnshock[case1] = jumpreturnshockmean[case1] + jumpreturnshockvol[case1] * scipy.stats.norm.ppf(ppfarg)

        returnshock[case2] = nojumpreturnshock[case2]

        ppfarg = (u[case3] - oneminuscondjumpprobability[case3]) / condjumpprobability[case3]
        returnshock[case3] = jumpreturnshockmean[case3] + jumpreturnshockvol[case3] * scipy.stats.norm.ppf(ppfarg)
        
        return returnshock
    
    @vectorised
    def __samplenextstate(self, state, returnshock):
        xi = self.__randomstate.normal(size=np.shape(state))
        return self.__params.meanlogvar * self.__oneminuspersistence + \
                self.__params.persistence * state + \
                self.__params.voloflogvar * self.__params.cor * returnshock + \
                self.__params.voloflogvar * np.sqrt(1. - self.__params.cor*self.__params.cor) * xi
    
    @vectorised                
    def __condsample(self, state, observation):
        assert observation is not None
        expstate = np.exp(state)
        returnshock = self.__samplereturnshock(expstate, observation)
        return self.__samplenextstate(state, returnshock)
    
    @vectorised
    def __uncondsample(self, state):
        logvarshock = self.__randomstate.normal(size=np.shape(state))
        return self.__params.meanlogvar * self.__oneminuspersistence + \
            self.__params.persistence * state + \
            self.__params.voloflogvar * logvarshock

    @vectorised
    def sample(self, state, stochfilter):
        state = npu.tondim2(state, True)
        if stochfilter.lastobservation is None:
            nextstate = self.__uncondsample(state)
        else:
            nextstate = self.__condsample(state, stochfilter.lastobservation)
        return nextstate

class SVL2LogVarTransitionDistribution(object):
    def __init__(self, params, randomstate=None):
        self.__params = params
        self.__randomstate = npu.randomstate() if randomstate is None else randomstate
        self.__meanlogvartimesoneminuspersistence = self.__params.meanlogvar * (1. - self.__params.persistence)
        
    @vectorised
    def sample(self, state, stochfilter):
        state = npu.tondim2(state, ndim1tocol=True)
        logvarshock = self.__randomstate.normal(size=np.shape(state))
        nextstate = self.__meanlogvartimesoneminuspersistence + self.__params.persistence * state + self.__params.voloflogvar * logvarshock
        stochfilter.context['logvarshock'] = logvarshock
        return nextstate

class WCSVLLogVarTransitionDistribution(object):
    def __init__(self, params, randomstate=None):
        self.__params = params
        self.__randomstate = npu.randomstate() if randomstate is None else randomstate
        self.__oneminuspersistence = (1. - self.__params.persistence)
        self.__meanlogvartimesoneminuspersistence = self.__params.meanlogvar * self.__oneminuspersistence
        self.__vc = self.__params.voloflogvar * self.__params.cor
        self.__scalar = self.__params.voloflogvar * np.sqrt(1. - self.__params.cor * self.__params.cor)
        
    @vectorised
    def sample(self, state, stochfilter):
        dt = stochfilter.context['dt']
        state = npu.tondim2(state, ndim1tocol=True)
        lastobservation = stochfilter.lastobservation if stochfilter.lastobservation is not None else 0.
        logvarshock = self.__randomstate.normal(size=np.shape(state))
        nextstate = dt * self.__meanlogvartimesoneminuspersistence + \
                (1. - dt * self.__oneminuspersistence) * state + \
                self.__vc * lastobservation * np.exp(-.5 * state) + \
                np.sqrt(dt) * self.__scalar * logvarshock
        return nextstate

class SVLJWeightingFunction(object):
    PI_TIMES_2 = math.pi * 2.
    
    def __init__(self, params):
        self.__jumpintensity = params.jumpintensity
        self.__oneminusjumpintensity = 1. - params.jumpintensity
        self.__jumpvol = params.jumpvol
    
    @vectorised
    def __call__(self, observation, particle, stochfilter):
        expparticle = np.exp(particle)
        observationsquared = observation * observation
        jumpvar = self.__jumpvol * self.__jumpvol
        temp = expparticle + jumpvar
        return self.__oneminusjumpintensity / np.sqrt(SVLJWeightingFunction.PI_TIMES_2 * expparticle) * np.exp(-.5 * observationsquared / expparticle) + \
                self.__jumpintensity / np.sqrt(SVLJWeightingFunction.PI_TIMES_2 * temp) * np.exp(-.5 * observationsquared / temp)

class SVL2WeightingFunction(object):
    PI_TIMES_2 = math.pi * 2.

    def __init__(self, params):
        self.__cor = params.cor
        self.__oneminusrhosquared = 1. - params.cor * params.cor
        self.__halfvoloflogvar = .5 * params.voloflogvar
    
    @vectorised
    def __call__(self, observation, particle, stochfilter):
        eta = stochfilter.context['logvarshock'][stochfilter.currentparticleidx, :]
        mean = np.exp(.5*particle) * self.__cor * (eta - self.__halfvoloflogvar)
        var = self.__oneminusrhosquared * np.exp(particle)
        return 1. / np.sqrt(SVL2WeightingFunction.PI_TIMES_2 * var) * np.exp(-.5 * (observation - mean) * (observation - mean) / var)
    
class WCSVLWeightingFunction(object):
    PI_TIMES_2 = math.pi * 2.
    
    def __init__(self, params):
        pass
    
    @vectorised
    def __call__(self, observation, particle, stochfilter):
        dt = stochfilter.context['dt']
        expparticle = np.exp(particle)
        var = dt * expparticle
        return 1. / np.sqrt(WCSVLWeightingFunction.PI_TIMES_2 * var) * np.exp(-.5 * observation * observation / var)

class SVLJPredictedObservationSampler(object):
    def __init__(self, params, randomstate=None):
        self.__randomstate = npu.randomstate() if randomstate is None else randomstate

    @vectorised    
    def __call__(self, priorparticle, stochfilter):
        return self.__randomstate.normal(scale=np.exp(.5 * priorparticle))

class SVL2PredictedObservationSampler(object):
    def __init__(self, params, randomstate=None):
        self.__params = params
        self.__randomstate = npu.randomstate() if randomstate is None else randomstate

    @vectorised
    def __call__(self, priorparticle, stochfilter):
        eta = stochfilter.context['logvarshock']
        xi = self.__randomstate.normal(size=np.shape(eta))
        return (self.__params.cor * eta + np.sqrt(1. - self.__params.cor*self.__params.cor) * xi - .5 * self.__params.cor * self.__params.voloflogvar) * np.exp(.5 * priorparticle)

class WCSVLPredictedObservationSampler(object):
    def __init__(self, params, randomstate=None):
        self.__params = params
        self.__randomstate = npu.randomstate() if randomstate is None else randomstate

    @vectorised    
    def __call__(self, priorparticle, stochfilter):
        dt = stochfilter.context['dt']
        return self.__randomstate.normal(scale=np.sqrt(dt) * np.exp(.5 * priorparticle))
