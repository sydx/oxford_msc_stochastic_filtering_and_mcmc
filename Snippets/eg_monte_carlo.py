import math
import sys

import matplotlib.pyplot as plt

import monte_carlo as mc
import processes as proc
import simulation as sim
import utilities as util

def plotApproximations_(ax, title, method, tss, wss, drift, volatility, initvalue, tsexact, xsexact):
    ax.plot(tsexact, xsexact, 'k-', label='exact')
    for i, (tsi, wsi) in enumerate(zip(tss, wss)):
        xsapprox = method(drift, volatility, tsi, wsi, initvalue)
        c = float(i) / float(len(tss) - 1) * 0.8
        N = len(tsi) - 1
        ax.plot(tsi, xsapprox, '-', color='c', alpha=1.-c, label='$N=%d$' % N)
    ax.set_title(title)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$\\hat{X}_t')
    ax.legend(loc=(1.03, 0.2))

def plotApproximations(pctdrift, pctvolatility, initvalue, ts, methods, coarsencount):
    drift = lambda x: pctdrift * x
    volatility = lambda x: pctvolatility * x

    ws = proc.generateBrownianMotion(ts)

    xsexact = proc.generateGeometricBrownianMotion(pctdrift, pctvolatility, ts, ws, initvalue)

    tss = [ts]
    wss = [ws]

    for i in xrange(coarsencount):
        tss.append(util.coarsen(tss[-1]))
        wss.append(util.coarsen(wss[-1]))

    for name, method in methods.iteritems():
        fig = plt.figure()
        ax = plt.axes([0.1, 0.1, 0.69, 0.8])
        plotApproximations_(ax, name, method, tss, wss, drift, volatility, initvalue, ts, xsexact)

def plotErrors(pctdrift, pctvolatility, initvalue, ts, methods, coarsencount, pathcount):
    drift = lambda x: pctdrift * x
    volatility = lambda x: pctvolatility * x

    errs = {}
    Ns = set()

    for k in xrange(pathcount):
        ws = proc.generateBrownianMotion(ts)

        tss = [ts]
        wss = [ws]

        for i in xrange(coarsencount):
            tss.append(util.coarsen(tss[-1]))
            wss.append(util.coarsen(wss[-1]))

        xsexact = proc.generateGeometricBrownianMotion(pctdrift, pctvolatility, ts, ws, initvalue)
        for i, (tsi, wsi) in enumerate(zip(tss, wss)):
            N = len(tsi) - 1
            Ns.add(N)
            for name, method in methods.iteritems():
                xsapprox = method(drift, volatility, tsi, wsi, initvalue)
                if name not in errs: errs[name] = {}
                if N not in errs[name]: errs[name][N] = 0.
                errs[name][N] += math.pow(xsexact[-1] - xsapprox[-1], 2.)

    Ns = sorted(Ns)

    fig = plt.figure()
    ax = plt.subplot(111)
    for name in methods:
        ax.plot(Ns, [errs[name][N] for N in Ns], 'o-', label=name)
    ax.set_ylabel('Mean square error')
    ax.set_xlabel('$N$')
    ax.legend()

def main():
    methods = {
        'Euler-Maruyama': mc.eulerMaruyama,
        'Milstein': mc.milstein
    }

    maxtime = 1.
    timecount = 128+1
    pctdrift = .3
    pctvolatility = .5
    initvalue = 1.
    coarsencount = 5
    pathcount = 100

    ts = sim.generateEquallySpacedTimes(maxtime, timecount)

    plotApproximations(pctdrift, pctvolatility, initvalue, ts, methods, coarsencount)
    plotErrors(pctdrift, pctvolatility, initvalue, ts, methods, coarsencount, pathcount)

    plt.show()

if __name__ == '__main__':
    sys.exit(main())
