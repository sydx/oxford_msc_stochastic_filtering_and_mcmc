from cStringIO import StringIO
import struct
import sys

import plotting

import matplotlib.pyplot as plt

def binary64ToBinary64BitStr(number):
    return ''.join(bin(ord(c)).replace('0b', '').rjust(8, '0') for c in struct.pack('!d', number))

def binary64ToBinary64BitStrWithCTypes(number):
    import ctypes
    return bin(ctypes.c_ulonglong.from_buffer(ctypes.c_double(number)).value).replace('0b', '')

def binary64BitListToBinary64BitStr(binary64BitList):
    return ''.join([str(x) for x in reversed(binary64BitList)])

def byteToBinary64BitList(byte):
    result = [0] * 8
    for i in xrange(8):
        result[i] = byte & 1
        byte >>= 1
    return result

def binary64ToBinary64BitList(number):
    result = [0] * 64
    for i, c in enumerate(struct.pack('!d', number)):
        start = 8 * (7 - i)
        end = start + 8
        result[start:end] = byteToBinary64BitList(ord(c))
    return result

def binary64ToBinary64BitListWithCTypes(number):
    import ctypes
    result = [0] * 64
    ulonglong = ctypes.c_ulonglong.from_buffer(ctypes.c_double(number)).value
    for i in xrange(64):
        result[i] = ulonglong & 1
        ulonglong >>= 1
    return result

def binary64BitListToFields(binary64BitList):
    sign = binary64BitList[63]
    exponent = binary64BitList[52:63]
    significand = binary64BitList[0:52]
    return {'sign': sign, 'exponent': exponent, 'significand': significand}

def prettyPrintBinary64BitList(binary64BitList):
    fields = binary64BitListToFields(binary64BitList)
    sign = fields['sign']
    exponent = fields['exponent']
    significand = fields['significand']
    s = StringIO()
    s.write('+-+-----------+----------------------------------------------------+\n')
    s.write('|s|exponent   |significand                                         |\n')
    s.write('+-+-----------+----------------------------------------------------+\n')
    s.write('|%s|%s|%s|\n' % (
            sign,
            ''.join([str(x) for x in reversed(exponent)]),
            ''.join([str(x) for x in reversed(significand)])))
    s.write('+-+-----------+----------------------------------------------------+\n')
    s.write('| |  6        | 5         4         3         2         1         0|\n')
    s.write('|3|21098765432|1098765432109876543210987654321098765432109876543210|\n')
    s.write('+-+-----------+----------------------------------------------------+\n')
    return s.getvalue()

def binary64BitListToBinary64(binary64BitList):
    fields = binary64BitListToFields(binary64BitList)
    sign = fields['sign']
    exponent = fields['exponent']
    significand = fields['significand']
    e = sum([x * 2**i for i, x in enumerate(exponent)])
    fraction = (-1.)**sign * float(1 + sum([significand[52-i] * 2**(-i) for i in xrange(1, 53)])) * 2**(e-1023)
    return fraction

def getNumberCounts(beta, p, emin, emax, normalised=False):
    numberCounts = {}

    for exponent in xrange(emin, emax+1):
        if normalised:
            d0min = beta - 1
        else:
            d0min = 0

        significand = [0] * p
        for d0 in xrange(d0min, beta):
            significand[0] = d0
            _getNumberCounts(1, significand, beta, p, exponent, numberCounts)

    return numberCounts

def _getNumberCounts(digitIndex, significand, beta, p, exponent, numberCounts):
    if digitIndex == p:
        number = 0.0
        for i, d in enumerate(significand):
            number += d*(beta**(-i))
        number *= beta**exponent
        if number not in numberCounts:
            numberCounts[number] = 1
            numberCounts[-number] = 1
        else:
            numberCounts[number] += 1
            numberCounts[-number] += 1
    else:
        for d in xrange(0, beta):
            significand[digitIndex] = d
            _getNumberCounts(digitIndex+1, significand, beta, p, exponent, numberCounts)
