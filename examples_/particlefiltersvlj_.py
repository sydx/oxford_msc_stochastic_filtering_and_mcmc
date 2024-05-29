import numpy as np

import filtering.particle
import filtering.run
import maths.numpyutils as npu
import sv.filtering.particle
import sv.generation

randomstate = npu.randomstate()
randomstate.seed(45)

from svparams import params_pitt2014_fig1

timecount = 2001  # 0, 1, ..., 1000

rmse = []
loglikelihood = []
effectivesamplesize = []

for iteration in range(20):
    print('Iteration %d...' % iteration)
    randomstate.seed()
    
    generator = sv.generation.SVDataGenerator(
            timecount=timecount,
            params=params_pitt2014_fig1,
            cortiming=sv.CorTiming.coratsametime,
            logreturnforward=False,
            logreturnscale=100.,
            randomstate=randomstate,
            usestratonovichcorrection=False)
    svdata = generator.generate()
    
    initialdistribution = sv.generation.LogVarInitialDistribution(params_pitt2014_fig1, randomstate)
    transitiondistribution = sv.filtering.particle.SVLJLogVarTransitionDistribution(params_pitt2014_fig1, randomstate)
    weightingfunction = sv.filtering.particle.SVLJWeightingFunction(params_pitt2014_fig1)    
    particlecount = 2000
    predictedobservationsampler = sv.filtering.particle.SVLJPredictedObservationSampler(params_pitt2014_fig1, randomstate)
    stochfilter = filtering.particle.RegularisedResamplingParticleFilter(
            initialdistribution=initialdistribution,
            transitiondistribution=transitiondistribution,
            weightingfunction=weightingfunction,
            particlecount=particlecount,
            statedim=1,
            observationdim=1,
            randomstate=randomstate,
            predictedobservationsampler=predictedobservationsampler)
    filterrundata = filtering.run.runfilter(svdata.svdf, params_pitt2014_fig1, stochfilter, 'logreturn', 'logvar')
    summary = filterrundata.summary()
    
    rmse.append(summary['rmse'])
    loglikelihood.append(summary['log-likelihood'])
    effectivesamplesize.append(summary['mean effective sample size'])

    print('rmse', np.mean(rmse), np.sqrt(np.var(rmse)))
    print('log-likelihood', np.mean(loglikelihood), np.sqrt(np.var(loglikelihood)))
    print('effective sample size', np.mean(effectivesamplesize), np.sqrt(np.var(effectivesamplesize)))
