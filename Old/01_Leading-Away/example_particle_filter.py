import numpy as np
import pylab

from euler_maruyama import eulerMaruyama
from particle_filter import ParticleFilter

# ------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------

# Time:                 [-----------------------------------------------]
#                       0.0                                             T
#
# Delta intervals:      [-----------------------|-----------------------]
#                       <----------------------->
#                                 delta
#
# Delta sub-intervals:  [-----------|-----------|-----------|-----------]
#                       <-----------> 
#                          delta/m
#
# Generation intervals: [---|---|---|---|---|---|---|---|---|---|---|---]

T = 2.0
deltaIntervalsCount = 8
deltaSubIntervalsPerDeltaIntervalCount = 10
m = deltaSubIntervalsPerDeltaIntervalCount
generationIntervalPerDeltaSubIntervalCount = 10
stateDim = 2
varDim = 2
numberOfParticles = 1000

delta = T / deltaIntervalsCount
generationIntervalCount = deltaIntervalsCount * \
    deltaSubIntervalsPerDeltaIntervalCount * \
    generationIntervalPerDeltaSubIntervalCount

# ------------------------------------------------------------------------------
# Generate the signal process
# ------------------------------------------------------------------------------

# Let's try the constant model first:
f = lambda  t, X : np.array([
    [0.5],
    [0.0]])
sigma = lambda t, X : np.array([
    [1.0, 0.0],
    [0.0, 1.0]])
x0 = np.array([
    [1.0],
    [2.0]])

times = np.linspace(0.0, T, generationIntervalCount + 1)

X = eulerMaruyama(f, sigma, x0, T, generationIntervalCount, varDim)

pylab.plot(times, X.T, '-')

# ------------------------------------------------------------------------------
# Generate the observation process
# ------------------------------------------------------------------------------

h = lambda t, X : X.copy()
variates = np.random.normal(loc=0.0, scale=1.0, size=(m, generationIntervalCount + 1))

# TODO: Apply h to columns:
# Y = np.cumsum(X, axis=1) * (T/generationIntervalCount) + np.cumsum(variates, axis=1) * np.sqrt(T/generationIntervalCount)
# Let's NOT add the observation noise (debugging):
Y = np.cumsum(X, axis=1) * (T/generationIntervalCount)
Y = np.roll(Y, 1, axis=1)
Y[:,0] = 0.0

print 'X:'
print X

print 'Y:'
print Y

# pylab.plot(times, Y.T, 'x')

# ------------------------------------------------------------------------------
# Initialise the particle filter
# ------------------------------------------------------------------------------

def initSampler():
    return x0.copy()
def observer(t):
    # print 'observer called with t=', t
    return Y[:,[t/(T/generationIntervalCount)]]

pf = ParticleFilter(
    numberOfParticles,
    delta,
    m,
    f,
    sigma,
    varDim,
    initSampler,
    observer,
    h,
    saveMeanHistory=True)
print 'starting!'
for i in xrange(deltaIntervalsCount):
    print 'evolving...'
    pf.evolve()
    print 'branching...'
    pf.branch()
    print 'got here:', pf.t
    pylab.axvline(pf.t)

print 'min(a):'
print np.min(pf.a)
print 'max(a):'
print np.max(pf.a)
print 'a:'
print pf.a
print 'v:'
print pf.v

print 'pf mean:'
print pf.getMean()

meanHistoryTimes = pf.getMeanHistoryTimes()
meanHistoryValues = pf.getMeanHistoryValues()

print np.shape(meanHistoryTimes)
print np.shape(meanHistoryValues)

pylab.plot(meanHistoryTimes, meanHistoryValues[0:])
pylab.show()
