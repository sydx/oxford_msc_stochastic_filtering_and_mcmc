from collections import namedtuple, OrderedDict
from timeit import default_timer as timer

import numpy as np
from pandas import DataFrame
from tabulate import tabulate

from maths.stats import OnlineMeanAndVarCalculator

class FilterRunData(namedtuple('FilterRunData', (
        'filterrundf',
        'params',
        'stochfilter',
        'duration'))):
    __slots__ = ()
    
    def summary(self):
        s = OrderedDict()
        s['stochastic filter'] = self.stochfilter
        if self.params is not None: 
            for param, value in self.params._asdict().items():
                s['parameter: %s' % param] = value
            s['log-variance theoretical half-life'] = self.params.logvarhalflife()
            s['log-variance theoretical unconditional s.d.'] = np.sqrt(self.params.logvaruncondvar())
        if 'rmse' in self.filterrundf.columns:
            s['rmse'] = self.filterrundf['rmse'].values[-1]
        elif 'truestate' in self.filterrundf.columns and 'posteriorstatemean' in self.filterrundf.columns:
            calculator = OnlineMeanAndVarCalculator()
            calculator.addall(self.filterrundf['truestate'].values - self.filterrundf['posteriorstatemean'].values)
            s['rmse'] = calculator.rms
        if 'loglikelihood' in self.filterrundf.columns:
            s['log-likelihood'] = self.filterrundf['loglikelihood'].values[-1]
        if 'effectivesamplesize' in self.filterrundf.columns:
            s['mean effective sample size'] = np.mean(self.filterrundf['effectivesamplesize'].values)
        s['duration in seconds'] = self.duration
        return s
            
    def __str__(self):
        rows = []
        for name, value in self.summary().items():
            rows.append((name, value))
        return tabulate(rows, headers=('item', 'value'))

def runfilter(df, params, stochfilter, observationcolname, truestatecolname, dropinitialrow=True, observationtransform=None, dtcolname=None):
    if dropinitialrow: df.drop(df.index[:1], inplace=True)
    
    cols = ['observation', 'posteriorstatemean', 'posteriorstatevar']
    havetruestate = truestatecolname is not None and \
            truestatecolname in df.columns and \
            np.any(df[truestatecolname].notnull().values)
    if havetruestate:
        cols.append('truestate')
        cols.append('error')
        cols.append('rmse')
        rmsecalculator = OnlineMeanAndVarCalculator()
    if hasattr(stochfilter, 'predictedobservation'): cols.append('predictedobservation')
    if hasattr(stochfilter, 'innovation'):
        cols.append('innovation')
        if hasattr(stochfilter, 'innovationvar'):
            cols.append('innovationvar')
            cols.append('standardisedinnovation')
    if hasattr(stochfilter, 'gain'):
        cols.append('gain')
        if havetruestate: cols.append('optimalgain')
    if hasattr(stochfilter, 'loglikelihood'):
        cols.append('loglikelihood')
    if hasattr(stochfilter, 'effectivesamplesize'):
        cols.append('effectivesamplesize')
        if hasattr(stochfilter, 'particlecount'):
            cols.append('effectivesamplesizethreshold')
    
    filterrundf = DataFrame(index=df.index, columns=cols)
    filterrundf.fillna(0.0, inplace=True)
    
    start = timer()
    
    for i, row in df.iterrows():
        print(i)
        if dtcolname is not None and dtcolname in df.columns:
            stochfilter.context['dt'] = row[dtcolname]
        else:
            stochfilter.context['dt'] = 1.
        
        stochfilter.predict()
        observation = row[observationcolname]
        
        if observationtransform is not None:
            observation = observationtransform(observation, stochfilter)
        
        stochfilter.observe(observation)
        filterrundf['observation'][i] = observation
        m = stochfilter.mean
        if (not np.isscalar(m)) and len(m) > 1: m = m[0,0]
        filterrundf['posteriorstatemean'][i] = m
        v = stochfilter.var
        if (not np.isscalar(v)) and len(v) > 1: v = v[0,0]
        filterrundf['posteriorstatevar'][i] = v
        if havetruestate:
            filterrundf['truestate'][i] = row[truestatecolname]
            error = row[truestatecolname] - m
            rmsecalculator.add(error)
            filterrundf['error'][i] = error
            filterrundf['rmse'][i] = rmsecalculator.rms
        if hasattr(stochfilter, 'predictedobservation'):
            filterrundf['predictedobservation'][i] = stochfilter.predictedobservation
        if hasattr(stochfilter, 'innovation'):
            filterrundf['innovation'][i] = stochfilter.innovation
            if hasattr(stochfilter, 'innovationvar'):
                filterrundf['innovationvar'][i] = stochfilter.innovationvar
                filterrundf['standardisedinnovation'][i] = stochfilter.innovation / np.sqrt(stochfilter.innovationvar)
        if hasattr(stochfilter, 'gain'):
            g = stochfilter.gain
            if (not np.isscalar(g)) and len(g) > 1: g = g[0,0]
            filterrundf['gain'][i] = g
            if havetruestate:
                filterrundf['optimalgain'][i] = (row[truestatecolname] - (m - g * stochfilter.innovation)) / stochfilter.innovation
        if hasattr(stochfilter, 'loglikelihood'):
            filterrundf['loglikelihood'][i] = stochfilter.loglikelihood
        if hasattr(stochfilter, 'effectivesamplesize'):
            filterrundf['effectivesamplesize'][i] = stochfilter.effectivesamplesize
            if hasattr(stochfilter, 'particlecount'):
                filterrundf['effectivesamplesizethreshold'][i] = 0.5 * float(stochfilter.particlecount)
                
    end = timer()
    
    return FilterRunData(
            filterrundf=filterrundf,
            params=params,
            stochfilter=stochfilter,
            duration = end - start)
