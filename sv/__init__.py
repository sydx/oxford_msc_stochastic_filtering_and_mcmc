from collections import namedtuple, OrderedDict
from enum import Enum

import numpy as np
from pandas import Series
from tabulate import tabulate

class CorTiming(Enum):
    # E.g. when log-return data loaded from a file, not generated:
    unknown = 0
    
    # Correlation between logvar[t+1] and logreturn[t]:
    coratsametime = 1
    
    # Correlation between logvar[t] and logreturn[t]:
    coronetimestepapart = 2

class SVData(namedtuple('SVData', (
        'sourcekind',
        'source',
        'svdf',
        'params',
        'cortiming',
        'logreturnforward',
        'logreturnscale'))):
    __slots__ = ()
    
    def summary(self):
        s = OrderedDict()
        s['source kind'] = self.sourcekind
        s['source'] = self.source
        if self.params is not None:
            for param, value in self.params._asdict().items():
                s['parameter: %s' % param] = value
            s['log-variance theoretical half-life'] = self.params.logvarhalflife()
            s['log-variance theoretical unconditional s.d.'] = np.sqrt(self.params.logvaruncondvar())
        s['log-return sample mean'] = np.mean(self.svdf['logreturn'])
        s['log-return sample s.d.'] = np.sqrt(np.var(self.svdf['logreturn']))
        if 'logvar' in self.svdf.columns:
            s['log-variance sample mean'] = np.mean(self.svdf['logvar'])
            s['log-variance sample s.d.'] = np.sqrt(np.var(self.svdf['logvar']))
        s['correlation timing'] = self.cortiming        
        s['log-return forward?'] = self.logreturnforward
        s['log-return scale'] = self.logreturnscale
        return s

    def __str__(self):
        rows = []
        for name, value in self.summary().items():
            rows.append((name, value))
        return tabulate(rows, headers=('item', 'value'))

class Params(namedtuple('Params', (
        'meanlogvar',
        'persistence',
        'voloflogvar',
        'cor',
        'jumpintensity',
        'jumpvol'))):
    __slots__ = ()
    
    def logvarhalflife(self):
        return np.log(0.5) / np.log(np.abs(self.persistence))
    
    def logvaruncondvar(self):
        return self.voloflogvar * self.voloflogvar / \
                (1. - self.persistence * self.persistence)
    
    def __str__(self):
        rows = []
        for param, value in self._asdict().items():
            rows.append((param, value))
        return tabulate(rows, headers=('parameter', 'value'))

def logreturntologprice(svdf, initialprice, logreturnisforward=False, logreturnscale=1.):
    logprices = np.empty(len(svdf) + 1)
    logprices[0] = np.log(initialprice)
    for i in range(len(svdf)):
        logreturn = svdf['logreturn'].iat[i]
        if np.isnan(logreturn): logreturn = 0.
        logreturn /= logreturnscale
        logprices[i+1] = logprices[i] + logreturn
    logprices = logprices[0:-1] if logreturnisforward else logprices[1:]
    return Series(logprices, index=svdf.index) 
