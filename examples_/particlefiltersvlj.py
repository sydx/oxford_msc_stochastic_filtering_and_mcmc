import matplotlib.pyplot as plt

import filtering.particle
import filtering.run
import filtering.visualisation
import maths.numpyutils as npu
import sv.filtering.particle
import sv.generation
import sv.visualisation

randomstate = npu.randomstate()
randomstate.seed(45)

from svparams import params_pitt2014_fig1

timecount = 2001  # 0, 1, ..., 1000

generator = sv.generation.SVDataGenerator(
        timecount=timecount,
        params=params_pitt2014_fig1,
        cortiming=sv.CorTiming.coratsametime,
        logreturnforward=False,
        logreturnscale=100.,
        randomstate=randomstate,
        usestratonovichcorrection=False)
svdata = generator.generate()

print(svdata)

fig = plt.figure()
sv.visualisation.makesvdataplot(fig, svdata)

initialdistribution = sv.generation.LogVarInitialDistribution(params_pitt2014_fig1, randomstate)
transitiondistribution = sv.filtering.particle.SVLJLogVarTransitionDistribution(params_pitt2014_fig1, randomstate)
weightingfunction = sv.filtering.particle.SVLJWeightingFunction(params_pitt2014_fig1)    
particlecount = 300
predictedobservationsampler = sv.filtering.particle.SVLJPredictedObservationSampler(params_pitt2014_fig1, randomstate)
stochfilter = filtering.particle.SmoothResamplingParticleFilter(
        initialdistribution=initialdistribution,
        transitiondistribution=transitiondistribution,
        weightingfunction=weightingfunction,
        particlecount=particlecount,
        statedim=1,
        observationdim=1,
        randomstate=randomstate,
        predictedobservationsampler=predictedobservationsampler)
filterrundata = filtering.run.runfilter(svdata.svdf, params_pitt2014_fig1, stochfilter, 'logreturn', 'logvar')

print(filterrundata)

fig = plt.figure()
filtering.visualisation.makeparticlehistogram(fig, filterrundata.stochfilter)

fig = plt.figure()
filtering.visualisation.makefilterrunplot(fig, filterrundata.filterrundf)        

plt.show()
