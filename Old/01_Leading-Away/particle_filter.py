import sys

import numpy as np

class ParticleFilter(object):

# ------------------------------------------------------------------------------
# Constructor
# ------------------------------------------------------------------------------

    def __init__(
            self,
            n,
            delta,
            m,
            f,
            sigma,
            signalCovDim,
            initSampler,
            observer,
            h,
            saveMeanHistory):
        self.n = n                     # Number of particles
        self.delta = delta             # Apply correction (branching) at times
                                       # [(k-1)*delta, k*delta]
        self.m = m                     # Divide the inter-branching intervals
                                       # into m subintervals of length delta/m
                                       # and apply the Euler method to generate
                                       # the trajectories of the particles

        self.deltaoverm = delta / m    # We'll be using these in calculations
        self.rootdeltaoverm = np.sqrt(self.deltaoverm)

        # Signal process specification:
        self.f = f
        self.sigma = sigma
        self.signalCovDim = signalCovDim

        self.observer = observer
        self.h = h

        # Now initialise the filter. This will set the particle positions,
        # self.v, and particle weights, self.a, and time, self.t
        self._initialise(initSampler)

        if saveMeanHistory:
            self._meanHistoryTimes = [self.t]
            self._meanHistoryValues = [self.getMean()]
        else:
            self._meanHistoryTimes = None
            self._meanHistoryValues = None

# ------------------------------------------------------------------------------
# Initialisation
# ------------------------------------------------------------------------------

    def _initialise(self, initSampler):
        self.v = [initSampler() for j in xrange(self.n)]
        self._resetWeights()
        self.sum_a = float(self.n)
        self.t = 0.0

    def _resetWeights(self):
        self.a = np.ones(self.n)

# ------------------------------------------------------------------------------
# Iteration
# ------------------------------------------------------------------------------

# 1. Evolution of the particles

    def evolve(self):
        for l in xrange(self.m):
            obsincr = self.observer(self.t + self.deltaoverm) - self.observer(self.t)
            newv = [None for j in xrange(self.n)]
            newb = [None for j in xrange(self.n)]
            newa = [None for j in xrange(self.n)]
            for j in xrange(self.n):
                # Generate the Gaussian random vector deltaV:
                deltaV = np.random.normal(loc=0.0, scale=1.0, size=(self.signalCovDim, 1))
                newv[j] = self.v[j] + \
                    self.f(self.t, self.v[j])*self.deltaoverm + \
                    np.dot(self.sigma(self.t, self.v[j]), deltaV*self.rootdeltaoverm)
                h_vj = self.h(self.t, self.v[j])
                norm_h_vj = np.linalg.norm(h_vj)
                term1 = np.inner(h_vj.flatten(), obsincr.flatten()) # * self.deltaoverm #      / self.deltaoverm  # Not / self.deltaoverm, it's * (time increment)
                term2 = 0.5 * self.deltaoverm * norm_h_vj * norm_h_vj
                newb[j] = term1 - term2
                newa[j] = self.a[j] * np.exp(newb[j])
            self.v = newv
            self.a = np.array(newa)
            self.t += self.deltaoverm

            self.sum_a = np.sum(self.a)

            if self._meanHistoryTimes is not None:
                self._meanHistoryTimes.append(self.t)
                self._meanHistoryValues.append(list(self.getMean()))

# 2. Branching procedure

    def _normaliseWeights(self):
        self.a /= self.sum_a

    def _calcNumberOfOffspring(self):
        o = np.empty(self.n)

        g = self.n
        h = self.n

        u = np.random.uniform(0.0, 1.0, self.n - 1)

        for j in xrange(self.n - 1):
            na = self.n * self.a[j]
            frac_na, floor_na = np.modf(na)
            frac_g, floor_g = np.modf(g)
            if frac_na + np.modf(g - na)[0] < 1.0:
                if frac_g == 0.0:
                    """
                    print 'DIVISION BY ZERO!'
                    print 'self.n', self.n
                    print 'self.a[j]', self.a[j]
                    print 'frac_na', frac_na
                    print 'np.modf(g - na)[0]', np.modf(g - na)[0]
                    print 'g', g
                    sys.exit()
                    """
                if u[j] < 1.0 - (frac_na / frac_g):
                    o[j] = floor_na
                else:
                    o[j] = floor_na + (h - floor_g)
            else:
                if u[j] < 1.0 - (1.0 - frac_na) / (1.0 - frac_g):
                    o[j] = floor_na + 1.0
                else:
                    o[j] = floor_na + (h - floor_g)
            g -= na
            h -= o[j]
        o[self.n - 1] = h

        return [int(x) for x in o]

    def branch(self):
        self._normaliseWeights()
        o = self._calcNumberOfOffspring()
        assert len(o) == self.n
        assert sum(o) == self.n

        childlessParticles = []
        for j in xrange(self.n):
            if o[j] == 0:
                childlessParticles.append(j)

        for j in xrange(self.n):
            for i in xrange(o[j] - 1):
                self.v[childlessParticles[-1]] = self.v[i]
                del childlessParticles[-1]

        self._resetWeights()

    def getMean(self):
        return np.average(self.v, axis=0, weights=self.a).flatten()

    def getMeanHistoryTimes(self):
        return np.array(self._meanHistoryTimes)

    def getMeanHistoryValues(self):
        return np.array(self._meanHistoryValues)
