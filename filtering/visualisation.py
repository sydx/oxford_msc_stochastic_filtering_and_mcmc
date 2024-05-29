from collections import OrderedDict

import numpy as np
import statsmodels.api as sm

def makefilterrunplot(fig, filterrundf):
    plotcount = 0
    if 'posteriorstatemean' in filterrundf.columns:
        plotcount += 1
        if 'error' in filterrundf.columns: plotcount += 1
        if 'rmse' in filterrundf.columns: plotcount += 1
        if 'posteriorstatevar' in filterrundf.columns: plotcount += 1
    if 'observation' in filterrundf.columns: plotcount += 1
    if 'innovation' in filterrundf.columns: plotcount += 1
    if 'standardisedinnovation' in filterrundf.columns: plotcount += 2
    if 'gain' in filterrundf.columns: plotcount += 1
    if 'loglikelihood' in filterrundf.columns: plotcount += 1
    if 'effectivesamplesize' in filterrundf.columns: plotcount += 1
    
    plots = OrderedDict()
    nextplotidx = 1
    
    if 'posteriorstatemean' in filterrundf.columns:
        stateplot = fig.add_subplot(plotcount, 1, nextplotidx)
        nextplotidx += 1
        if 'posteriorstatevar' in filterrundf.columns:
            sds = np.sqrt(filterrundf['posteriorstatevar'])
            stateplot.plot(filterrundf['posteriorstatemean'], 'b', label='posterior filtered state')
            stateplot.plot(filterrundf['posteriorstatemean'] + sds, 'b--') 
            stateplot.plot(filterrundf['posteriorstatemean'] - sds, 'b--')
        if 'truestate' in filterrundf.columns:
            stateplot.plot(filterrundf['truestate'], 'r', label='true state') 
        stateplot.set_title('state', fontsize=9)
        stateplot.legend(loc='upper right', fancybox=True, framealpha=0.5, prop={'size': 8})
        plots['stateplot'] = stateplot
        
        if 'error' in filterrundf.columns:
            errorplot = fig.add_subplot(plotcount, 1, nextplotidx)
            nextplotidx += 1
            errorplot.plot(filterrundf['error'], 'b')
            errorplot.set_title('error', fontsize=9)
            plots['errorplot'] = errorplot
    
        if 'rmse' in filterrundf.columns:
            rmseplot = fig.add_subplot(plotcount, 1, nextplotidx)
            nextplotidx += 1
            rmseplot.plot(filterrundf['rmse'], 'b')
            rmseplot.set_title('rmse', fontsize=9)
            plots['rmseplot'] = rmseplot
    
        if 'posteriorstatevar' in filterrundf.columns:
            statesdplot = fig.add_subplot(plotcount, 1, nextplotidx)
            nextplotidx += 1
            statesdplot.plot(sds, 'b')
            statesdplot.set_title('posterior filtered state standard deviation', fontsize=9)
            plots['statesdplot'] = statesdplot
            
    if 'observation' in filterrundf.columns:
        observationplot = fig.add_subplot(plotcount, 1, nextplotidx)
        nextplotidx += 1
        observationplot.plot(filterrundf['observation'], 'r', label='true observation')
        if 'predictedobservation' in filterrundf.columns:
            observationplot.plot(filterrundf['predictedobservation'], 'b', label='predicted observation')
            observationplot.legend(loc='upper right', fancybox=True, framealpha=0.5, prop={'size': 8})
        observationplot.set_title('observation', fontsize=9)
        plots['observationplot'] = observationplot
    
    if 'standardisedinnovation' in filterrundf.columns:
        innovationplot = fig.add_subplot(plotcount, 1, nextplotidx)
        nextplotidx += 1
        innovationplot.plot(filterrundf['innovation'], 'b', label='innovation')
        if 'innovationvar' in filterrundf.columns:
            innovationsds = np.sqrt(filterrundf['innovationvar'])
            innovationplot.plot(filterrundf['innovation'] + innovationsds, 'b--')
            innovationplot.plot(filterrundf['innovation'] - innovationsds, 'b--')
            innovationplot.legend(loc='upper right', fancybox=True, framealpha=0.5, prop={'size': 8})
        innovationplot.set_title('innovation', fontsize=9)
        plots['innovationplot'] = innovationplot
        
    if 'standardisedinnovation' in filterrundf.columns:
        innovationqqplot = fig.add_subplot(plotcount, 1, nextplotidx)
        nextplotidx += 1
        sm.qqplot(filterrundf['standardisedinnovation'].values, line='45', ax=innovationqqplot)
        innovationqqplot.set_title('standardised innovation Q-Q plot', fontsize=9)
        innovationqqplot.set_xlabel('')
        innovationqqplot.set_ylabel('')
        plots['innovationqqplot'] = innovationqqplot
    
        cusumplot = fig.add_subplot(plotcount, 1, nextplotidx)
        nextplotidx += 1
        maxtime = filterrundf.index[-1]
        meanstandardisedinnovation = np.mean(filterrundf['standardisedinnovation'].values)
        variance = np.sum(np.square(filterrundf['standardisedinnovation'].values - meanstandardisedinnovation))
        cusum = np.cumsum(filterrundf['standardisedinnovation'].values)
        cusum /= np.sqrt(variance)
        a = 0.850
        topline = a * np.sqrt(maxtime) + 2 * a * filterrundf.index / np.sqrt(maxtime)
        cusumplot.plot(cusum, 'b', label='CUSUM')
        #cusumplot.plot(topline, 'r', label='5% significance')
        #cusumplot.plot(-topline, 'r')
        cusumplot.set_title('CUSUM', fontsize=9)
        plots['cusumplot'] = cusumplot
        
    if 'gain' in filterrundf.columns:
        gainplot = fig.add_subplot(plotcount, 1, nextplotidx)
        nextplotidx += 1
        gainplot.plot(filterrundf['gain'], 'b', label='gain')
        if 'optimalgain' in filterrundf.columns:
            gainplot.plot(filterrundf['optimalgain'], 'r', label='optimal gain')
            gainplot.legend(loc='upper right', fancybox=True, framealpha=0.5, prop={'size': 8})
        gainplot.set_title('gain', fontsize=9)
        plots['gainplot'] = gainplot
        
    if 'loglikelihood' in filterrundf.columns:
        loglikelihoodplot = fig.add_subplot(plotcount, 1, nextplotidx)
        nextplotidx += 1
        loglikelihoodplot.plot(filterrundf['loglikelihood'], 'b', label='loglikelihood')
        loglikelihoodplot.set_title('loglikelihood', fontsize=9)
        plots['loglikelihoodplot'] = loglikelihoodplot
        
    if 'effectivesamplesize' in filterrundf.columns:
        effectivesamplesizeplot = fig.add_subplot(plotcount, 1, nextplotidx)
        nextplotidx += 1
        effectivesamplesizeplot.plot(filterrundf['effectivesamplesize'], 'b', label='effective sample size')
        if 'effectivesamplesizethreshold' in filterrundf.columns:
            effectivesamplesizeplot.plot(filterrundf['effectivesamplesizethreshold'], 'r', label='threshold')
        effectivesamplesizeplot.set_title('effective sample size', fontsize=9)
        plots['effectivesamplesizeplot'] = effectivesamplesizeplot
        
    fig.subplots_adjust(hspace=.5)
    plotidx = 0
    for plot in plots.values():
        if plotidx == len(plots) - 1: plot.set_xlabel('time', fontsize=8)
        plot.tick_params(axis='both', which='major', labelsize=7)
        plot.tick_params(axis='both', which='minor', labelsize=7)
        plotidx += 1
            
    return plots

def makeparticlehistogram(fig, particlefilter):
    plot = fig.add_subplot(111)
    # Each particle corresponds to a *row* in
    # particlefilter.resampledparticles. 2D hist input must be
    # nsamples x nvariables, each sample being a particle, so we are good, don't
    # have to transpose (.T).
    plot.hist(particlefilter.resampledparticles, bins=20)
    return plot

