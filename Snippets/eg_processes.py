import sys

import matplotlib.pyplot as plt

import framework
import processes as proc
import simulation as sim

def main(args):
    figures = {}

    ts = sim.generateEquallySpacedTimes(1., 100)
    ws = proc.generateBrownianMotion(ts, dim=1)
    xs = proc.generateCorrelatedBrownianMotionWithDrift([0.3, 0.4], [1.0, 3.0], ts, ws, 0.0)
    ws = proc.generateBrownianMotion(ts, dim=2)
    xs = proc.generateCorrelatedBrownianMotionWithDrift([0.3, 0.4], [[1.0, 0.0], [0.0, 3.0]], ts, ws, 1.0)
    xs = proc.generateCorrelatedBrownianMotionWithDrift([0.3, 0.4], [[1.0, 0.0], [0.0, 1.0]], ts, ws, [1.0, -2.0])

    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(ts, ws, label='W')
    ax.plot(ts, xs, label='X')
    plt.legend()
    figures['processes'] = fig

    framework.processFigures(figures, args)
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Example: processes')
    parser.add_argument('--figuresoutpath', help='save figures to files in specified directory')
    args = parser.parse_args()
    sys.exit(main(framework.parseCommandLineArguments()))
