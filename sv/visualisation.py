from collections import namedtuple, OrderedDict

import numpy as np

def makesvdataplot(fig, svdata):
    svdf = svdata.svdf

    plotcount = 0
    if 'logvar' in svdf.columns: plotcount += 3
    if 'logreturn' in svdf.columns: plotcount += 1
    if 'logprice' in svdf.columns: plotcount += 1
    if 'price' in svdf.columns: plotcount += 1

    plots = OrderedDict()
    nextplotidx = 1
    
    if 'logvar' in svdf.columns:    
        logvarplot = fig.add_subplot(plotcount, 1, nextplotidx)
        nextplotidx += 1
        logvarplot.plot(svdf['logvar'])
        logvarplot.set_title('log-variance')
        plots['logvarplot'] = logvarplot

        varplot = fig.add_subplot(plotcount, 1, nextplotidx)
        nextplotidx += 1
        var = np.exp(svdf['logvar'])
        varplot.plot(var)
        varplot.set_title('variance')
        plots['varplot'] = varplot
    
        sdplot = fig.add_subplot(plotcount, 1, nextplotidx)
        nextplotidx += 1
        sd = np.sqrt(var)
        sdplot.plot(sd)
        sdplot.set_title('volatility (standard deviation)')
        plots['sdplot'] = sdplot

    if 'logreturn' in svdf.columns:    
        logreturnplot = fig.add_subplot(plotcount, 1, nextplotidx)
        nextplotidx += 1
        logreturnplot.plot(svdf['logreturn'])
        logreturnplot.set_title('log-return')
        plots['logreturnplot'] = logreturnplot
        
    if 'logprice' in svdf.columns:
        logpriceplot = fig.add_subplot(plotcount, 1, nextplotidx)
        nextplotidx += 1
        logpriceplot.plot(svdf['logprice'])
        logpriceplot.set_title('log-price')
        plots['logpriceplot'] = logpriceplot

    if 'price' in svdf.columns:
        priceplot = fig.add_subplot(plotcount, 1, nextplotidx)
        nextplotidx += 1
        priceplot.plot(svdf['price'])
        priceplot.set_xlabel('time')
        priceplot.set_title('price')
        plots['priceplot'] = priceplot

    if 'jumpflag' in svdf.columns:        
        for i in svdf[svdf['jumpflag']].index:
            for plot in plots.values():
                plot.axvline(x=i, color='red')

    fig.subplots_adjust(hspace=.5)
    
    return plots
