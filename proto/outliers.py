import numpy as np
import statsmodels.api as sm

def problessthan(sample, bw, value, count, randomstate):
    idxs = randomstate.randint(0, len(sample), size=count)
    epsilons = randomstate.normal(size=count)
    flags = sample[idxs] + bw * epsilons < value
    return float(np.sum(flags))/float(count)

randomstate = np.random.RandomState(seed=42)
nobs = 300
sample = randomstate.normal(size=nobs)
kde = sm.nonparametric.KDEUnivariate(np.random.normal(size=nobs))
kde.fit()
print(problessthan(sample, kde.bw, 100., 10, randomstate)
