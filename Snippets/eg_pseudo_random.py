import sys

import numpy as np
import matplotlib.pyplot as plt

import framework

def latticeTest():
    size = 10000
    numbers = np.random.uniform(size=size)
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(numbers[0::2], numbers[1::2], '.')
    return fig

def histogramTest():
    mu, sigma = 0., 1.0
    variates = np.random.normal(mu, sigma, 10000)
    fig = plt.figure()
    ax = plt.subplot(111)
    count, bins, ignored = ax.hist(variates, 60, normed=True)
    ax.plot(bins, 1./(sigma * np.sqrt(2*np.pi)) * np.exp(-(bins-mu)**2 / (2*sigma**2)), linewidth=2, color='r')
    return fig

def main(args):
    figures = {}

    figures['lattice-test'] = latticeTest()
    figures['histogram-test'] = histogramTest()

    framework.processFigures(figures, args)
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Example: pseudo-random number generation')
    parser.add_argument('--figuresoutpath', help='save figures to files in specified directory')
    args = parser.parse_args()
    sys.exit(main(framework.parseCommandLineArguments()))
