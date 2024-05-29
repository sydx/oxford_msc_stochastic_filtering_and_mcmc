import struct
import sys

import floating_point
import plotting

import matplotlib.pyplot as plt

def binaryRepresentationOfPythonFloat():
    print 'sys.float_info', sys.float_info
    number = -3.578
    binary64BitList = floating_point.binary64ToBinary64BitList(number)
    print floating_point.binary64BitListToBinary64BitStr(binary64BitList)
    binary64BitList = floating_point.binary64ToBinary64BitListWithCTypes(number)
    print floating_point.binary64BitListToBinary64BitStr(binary64BitList)
    print floating_point.binary64ToBinary64BitStr(number)
    print floating_point.binary64ToBinary64BitStrWithCTypes(number)
    print floating_point.binary64BitListToBinary64(binary64BitList)
    print floating_point.prettyPrintBinary64BitList(binary64BitList)

def floatingPointRepresentation():
    numberCounts = floating_point.getNumberCounts(beta=2, p=3, emin=-1, emax=2, normalised=False)

    fig = plt.figure()
    ax = plt.subplot(111)
    for x in numberCounts:
        ax.axvline(x)
    ax.set_xlabel('real line')
    ax.set_yticks([])
    plotting.adjustFigAspect(fig, aspect=6.)

    fig = plt.figure()
    ax = plt.subplot(111)
    numbers = []
    counts = []
    for n in sorted(numberCounts):
        numbers.append(n)
        counts.append(numberCounts[n])
    ax.bar(numbers, counts, width=0.08, linewidth=0)
    ax.set_xlabel('real line')
    ax.set_ylabel('number of representations')

def normalisedFloatingPointRepresentation():
    numbers = sorted(floating_point.getNumberCounts(beta=2, p=3, emin=-1, emax=2, normalised=True))
    
    fig = plt.figure()
    ax = plt.subplot(111)
    for x in numbers:
        ax.axvline(x)
    ax.set_xlabel('real line')
    ax.set_yticks([])
    plotting.adjustFigAspect(fig, aspect=6.)

def main():
    binaryRepresentationOfPythonFloat()
    floatingPointRepresentation()
    normalisedFloatingPointRepresentation()
    plt.show()

if __name__ == '__main__':
    sys.exit(main())
