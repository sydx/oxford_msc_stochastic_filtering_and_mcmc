import numpy as np
import numpy.linalg as la

import scipy.stats as stats

import maths.numpypreconditions as npp
import maths.numpyutils as npu

class MultivariateNormalDistribution(object):
    def __init__(self, mean, covariance, randomstate=None):
        self.__mean = npu.immutablecopyof(npu.tondim1(mean))
        self.__covariance = npu.immutablecopyof(npp.checkshapeissquare(npu.tondim2(covariance)))
        self.__randomstate = npu.randomstate() if randomstate is None else randomstate
        self.__impl = stats.multivariate_normal(self.__mean, self.__covariance)

    def __getmean(self):
        return self.__mean

    mean = property(fget=__getmean)

    def __getcovariance(self):
        return self.__covariance

    covariance = property(fget=__getcovariance)

    def sample(self, size=1):
        return self.__randomstate.multivariate_normal(self.__mean, self.__covariance, size)

    def pdf(self, value):
        return self.__impl.pdf(value)

class MixtureDistributionElement(object):
    def __init__(self, weight, distribution):
        self.__weight = weight
        self.__distribution = distribution

    def __getweight(self):
        return self.__weight

    weight = property(fget=__getweight)

    def __getdistribution(self):
        return self.__distribution

    distribution = property(fget=__getdistribution)

class MixtureDistribution(object):
    def __init__(self, elements, randomstate=None):
        self.__elements = [e for e in elements]
        self.__randomstate = npu.randomstate() if randomstate is None else randomstate

    def __getitem__(self, idx):
        return self.__elements[idx]

    def sample(self, size=1):
        pvals = [e.weight for e in self.__elements]
        u = self.__randomstate.multinomial(1, pvals, size)

        result = []
        for sample in u:
            elementidx = np.ravel(np.where(sample))[0]
            result.append(self.__elements[elementidx].distribution.sample()[0])

        return np.array(result)

    def pdf(self, value):
        result = 0.
        for e in self.__elements:
            result += e.weight * e.distribution.pdf(value)
        return result

# The following are the six example distributions from
#
# @Article{doi:10.1080/10485250306039,
#   Title                    = {Plug-in bandwidth matrices for bivariate kernel density estimation},
#   Author                   = {Duong, Tarn and Hazelton, Martin},
#   Journal                  = {Journal of Nonparametric Statistics},
#   Year                     = {2003},
#   Number                   = {1},
#   Pages                    = {17-30},
#   Volume                   = {15},
#   Doi                      = {10.1080/10485250306039},
#   Eprint                   = {http://dx.doi.org/10.1080/10485250306039},
#   Url                      = {http://dx.doi.org/10.1080/10485250306039}
# }

mixturedistributionA = MixtureDistribution((
    MixtureDistributionElement(1., MultivariateNormalDistribution(np.array((0., 0.)), np.array(((.25, 0.), (0., 1.))))),
    ))
mixturedistributionB = MixtureDistribution((
    MixtureDistributionElement(.5, MultivariateNormalDistribution(np.array((1., 0.)), np.array((((4./9.), 0.), (0., (4./9.)))))),
    MixtureDistributionElement(.5, MultivariateNormalDistribution(np.array((-1., 0.)), np.array((((4./9.), 0.), (0., (4./9.)))))),
    ))
mixturedistributionC = MixtureDistribution((
    MixtureDistributionElement(.5, MultivariateNormalDistribution(np.array(((3./2.), 0.)), np.array((((1./16.), 0.), (0., 1.))))),
    MixtureDistributionElement(.5, MultivariateNormalDistribution(np.array(((-3./2.), 0.)), np.array((((1./16.), 0.), (0., 1.))))),
    ))
mixturedistributionD = MixtureDistribution((
    MixtureDistributionElement(.5, MultivariateNormalDistribution(np.array((1., -1.)), np.array((((4./9.), (14./45.)), ((14./45.), (4./9.)))))),
    MixtureDistributionElement(.5, MultivariateNormalDistribution(np.array((-1., 1.)), np.array((((4./9.), 0.), (0., (4./9.)))))),
    ))
mixturedistributionE = MixtureDistribution((
    MixtureDistributionElement((3./7.), MultivariateNormalDistribution(np.array((-1., 0.)), np.array((((9./25.), (63./250.)), ((63./250.), (49./100.)))))),
    MixtureDistributionElement((3./7.), MultivariateNormalDistribution(np.array((1., 2./np.sqrt(3.))), np.array((((9./25.), 0.), (0., (49./100.)))))),
    MixtureDistributionElement((1./7.), MultivariateNormalDistribution(np.array((1., -2./np.sqrt(3.))), np.array((((9./25.), 0.), (0., (49./100.)))))),
    ))
mixturedistributionF = MixtureDistribution((
    MixtureDistributionElement(1., MultivariateNormalDistribution(np.array((0., 0.)), np.array(((1., .9), (.9, 1.))))),
    ))
