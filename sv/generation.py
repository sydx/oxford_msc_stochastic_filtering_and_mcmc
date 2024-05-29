import numpy as np
import pandas as pd

import filtering.generation as gen
import maths.numpyutils as npu
import maths.randomness as rnd
from sv import CorTiming, SVData

class LogVarInitialDistribution(object):
    def __init__(self, params, randomstate=None):
        self.__scale = params.voloflogvar / np.sqrt(1. - params.persistence * params.persistence)
        randomstate = npu.randomstate() if randomstate is None else randomstate
        self.__normalvariatesgenerator = rnd.NormalVariatesGenerator(randomstate)
        
    def sample(self, normalvariate=None):
        if normalvariate is None:
            normalvariate = self.__normalvariatesgenerator.generatenormalvariates(1., 1)
        normalvariate = npu.toscalar(normalvariate)
        return self.__scale * normalvariate
    
class SVDataGenerator(object):
    def __init__(self, timecount, params, cortiming, logreturnforward, logreturnscale, randomstate=None, usestratonovichcorrection=False):
        self.__timecount = timecount
        self.__params = params
        self.__cortiming = cortiming
        self.__logreturnforward = logreturnforward
        self.__logreturnscale = logreturnscale
        self.__randomstate = npu.randomstate() if randomstate is None else randomstate
        self.__logvarinitialdistribution = LogVarInitialDistribution(params, self.__randomstate)
        self.__usestratonovichcorrection = usestratonovichcorrection
        
    def generate(self):
        initialprice = 100.
        
        initiallogvar = self.__logvarinitialdistribution.sample()
        
        constterm = self.__params.meanlogvar * (1. - self.__params.persistence) 
        
        generator = gen.StateSpaceModelDataGenerator(self.__randomstate)
        generator.addnoise('epsilon', var=1., lag=0)
        etalag = 1 if self.__cortiming == CorTiming.coratsametime else 0
        generator.addnoise('eta', var=1., lag=etalag)
        generator.setnoisecors((self.__params.cor,))
        generator.addjump('jump', intensity=self.__params.jumpintensity, sd=self.__params.jumpvol)
        
        stratonovichcorrection = -.5 * self.__params.cor * self.__params.voloflogvar if self.__usestratonovichcorrection else 0.

        def logvar(time, processname, data):
            if time == 0: return initiallogvar
            return constterm + \
                    self.__params.persistence * data.process('logvar', time - 1) + \
                    self.__params.voloflogvar * data.noise('eta', time)
        def logreturn(time, processname, data):
            if time == 0: return np.nan
            return (data.noise('epsilon', time) + stratonovichcorrection) * \
                    np.exp(.5 * data.process('logvar', time)) + \
                    data.jump('jump', time)
        def logpricefrombackwardlogreturn(time, processname, data):
            prevlogprice = np.log(initialprice) if time == 0 else data.process('logprice', time - 1)
            logreturn = data.process('logreturn', time)
            return prevlogprice if np.isnan(logreturn) else prevlogprice + logreturn / self.__logreturnscale
        def logpricefromforwardlogreturn(time, processname, data):
            if time == 0: return np.log(initialprice)
            prevlogprice = data.process('logprice', time - 1)
            prevlogreturn = data.process('logreturn', time - 1)
            if np.isnan(prevlogreturn): prevlogreturn = 0.
            return prevlogprice + prevlogreturn / self.__logreturnscale
        def price(time, processname, data):
            return np.exp(data.process('logprice', time))
         
        generator.addprocess('logvar', logvar)
        generator.addprocess('logreturn', logreturn)
        logprice = logpricefromforwardlogreturn if self.__logreturnforward else logpricefrombackwardlogreturn
        generator.addprocess('logprice', logprice)
        generator.addprocess('price', price)
        generator.settimecount(self.__timecount)
        data = generator.generate()
        processdf = data.processdf()
        jumpflagdf = data.jumpflagdf(colname=lambda idx, name: '%sflag' % name)
        jumpdf = data.jumpdf()
        svdf = pd.concat((processdf, jumpflagdf, jumpdf), axis=1)
        return SVData(
                sourcekind='generator',
                source=self,
                svdf=svdf,
                params=self.__params,
                cortiming=self.__cortiming,
                logreturnforward=self.__logreturnforward,
                logreturnscale=self.__logreturnscale)
        