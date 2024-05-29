import matplotlib.pyplot as plt
import numpy as np

import filtering.particle
import filtering.run
import filtering.visualisation
import maths.numpyutils as npu
import sv.filtering.particle
import sv.generation
import sv.visualisation

randomstate = npu.randomstate()
randomstate.seed(42)

from svparams import params_pitt2014_fig1

timecount = 2001  # 0, 1, ..., 1000

rmse = []
loglikelihood = []
effectivesamplesize = []
plot = False

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
    
    if plot:
        fig = plt.figure()
        sv.visualisation.makesvdataplot(fig, svdata)
    
    newparams = sv.Params(
        meanlogvar    = 0.25,
        persistence   = 0.975,
        cor           = -0.8,
        voloflogvar   = np.sqrt(0.025),
        jumpintensity = 0.0,
        jumpvol       = 10.)

    outlierthreshold = .05
    
    initialdistribution = sv.generation.LogVarInitialDistribution(newparams, randomstate)
    transitiondistribution = sv.filtering.particle.SVLJLogVarTransitionDistribution(newparams, randomstate)
    weightingfunction = sv.filtering.particle.SVLJWeightingFunction(newparams)    
    particlecount = 300
    predictedobservationsampler = sv.filtering.particle.SVLJPredictedObservationSampler(newparams, randomstate)
    stochfilter = filtering.particle.SmoothResamplingParticleFilter(
            initialdistribution=initialdistribution,
            transitiondistribution=transitiondistribution,
            weightingfunction=weightingfunction,
            particlecount=particlecount,
            statedim=1,
            observationdim=1,
            randomstate=randomstate,
            predictedobservationsampler=predictedobservationsampler,
            outlierthreshold=outlierthreshold)
    try:
        filterrundata = filtering.run.runfilter(svdata.svdf, newparams, stochfilter, 'logreturn', 'logvar')
    except:
        continue
    summary = filterrundata.summary()
    
    rmse.append(summary['rmse'])
    loglikelihood.append(summary['log-likelihood'])
    effectivesamplesize.append(summary['mean effective sample size'])

    print('count', len(rmse))
    print('rmse', np.mean(rmse), np.sqrt(np.var(rmse)))
    print('log-likelihood', np.mean(loglikelihood), np.sqrt(np.var(loglikelihood)))
    print('effective sample size', np.mean(effectivesamplesize), np.sqrt(np.var(effectivesamplesize)))

    if plot:
        fig = plt.figure()
        filtering.visualisation.makeparticlehistogram(fig, filterrundata.stochfilter)
    
        fig = plt.figure()
        filtering.visualisation.makefilterrunplot(fig, filterrundata.filterrundf)
        
        plt.show()        
