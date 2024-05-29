import os

import matplotlib.pyplot as plt
import numpy as np

import filtering.particle
import filtering.run
import filtering.visualisation
import maths.numpyutils as npu
import mcmc.bugs
import sv.filtering.particle
import sv.generation
import sv.loading
import sv.visualisation

import svparams

"""
Dataset 2

SVL

loglikelihood -2715.33216892
logparamposterior 9.66885573625
difference -2725.00102466

SVL2

loglikelihood -2721.33559092
logparamposterior 9.73037930714
difference -2731.06597023

>>> np.exp(-2725.00102466 - (-2731.06597023))
430.49924496056474 (svl over svl2)

Dataset 1

SVL

loglikelihood -922.985746553
logparamposterior 7.72786421116
difference -930.713610764

SVL2

loglikelihood -918.286532854
logparamposterior 7.73668424127
difference -926.023217096

>>> np.exp(-926.023217096 - (-930.713610764))
108.8960402568398 (svl2 over svl)
"""
def runsvljparticlefilter(svdata, params, randomstate):
    initialdistribution = sv.generation.LogVarInitialDistribution(params, randomstate)
    transitiondistribution = sv.filtering.particle.SVLJLogVarTransitionDistribution(params, randomstate)
    weightingfunction = sv.filtering.particle.SVLJWeightingFunction(params)    
    particlecount = 600
    predictedobservationsampler = sv.filtering.particle.SVLJPredictedObservationSampler(params, randomstate)
    stochfilter = filtering.particle.MultinomialResamplingParticleFilter(
            initialdistribution=initialdistribution,
            transitiondistribution=transitiondistribution,
            weightingfunction=weightingfunction,
            particlecount=particlecount,
            statedim=1,
            observationdim=1,
            randomstate=randomstate,
            predictedobservationsampler=predictedobservationsampler)
    return filtering.run.runfilter(svdata.svdf, params, stochfilter, 'logreturn', 'logvar')
    
def runsvl2particlefilter(svdata, params, randomstate):
    initialdistribution = sv.generation.LogVarInitialDistribution(params, randomstate)
    transitiondistribution = sv.filtering.particle.SVL2LogVarTransitionDistribution(params, randomstate)
    weightingfunction = sv.filtering.particle.SVL2WeightingFunction(params)    
    particlecount = 600
    predictedobservationsampler = sv.filtering.particle.SVL2PredictedObservationSampler(params, randomstate)
    stochfilter = filtering.particle.MultinomialResamplingParticleFilter(
            initialdistribution=initialdistribution,
            transitiondistribution=transitiondistribution,
            weightingfunction=weightingfunction,
            particlecount=particlecount,
            statedim=1,
            observationdim=1,
            randomstate=randomstate,
            predictedobservationsampler=predictedobservationsampler)
    return filtering.run.runfilter(svdata.svdf, params, stochfilter, 'logreturn', 'logvar')

rootdir = r'C:\Users\Paul\Documents\dev\alexandria\bilokon-msc\dissertation\code'

model = 'svl2'
ds = 'dataset-13'
params = svparams.params[(model, ds)]

resultsdir = os.path.join(rootdir, 'results', 'mcmc')
datasetsdir = os.path.join(rootdir, 'datasets')
codaindexfile = os.path.join(resultsdir, model, ds, 'coda-index.txt')
codadatafile = os.path.join(resultsdir, model, ds, 'coda-chain-1.txt')
observationsfile = os.path.join(datasetsdir, '%s_y.txt' % ds)

bc = mcmc.bugs.BUGSChain(codaindexfile, codadatafile)
bc.load()
nodenames = ['mu', 'phi', 'rho', 'sigmav']
print('Considering ONLY the parameters:', nodenames)
kde = bc.kde(nodenames)
print('KDE bandwidths for these parameters:', kde.bw)
paramposterior = kde.pdf([params.meanlogvar, params.persistence, params.cor, params.voloflogvar])
logparamposterior = np.log(paramposterior)

print('Loading SV data...')
svdata = sv.loading.loadSVDataFromBUGSDataset(observationsfile, logreturnforward=True, logreturnscale=100.)
print(svdata)

fig = plt.figure()
sv.visualisation.makesvdataplot(fig, svdata)

randomstate = npu.randomstate()
if model == 'svl':
    filterrundata = runsvljparticlefilter(svdata, params, randomstate)
elif model == 'svl2':
    filterrundata = runsvl2particlefilter(svdata, params, randomstate)
print(filterrundata)

fig = plt.figure()
filtering.visualisation.makeparticlehistogram(fig, filterrundata.stochfilter)
fig = plt.figure()
filtering.visualisation.makefilterrunplot(fig, filterrundata.filterrundf)        

loglikelihood = filterrundata.summary()['log-likelihood']

print('log-likelihood:', loglikelihood)
print('log of params posterior:', logparamposterior)
print('difference (for computing the Bayes factors):', loglikelihood - logparamposterior) 

plt.show()
