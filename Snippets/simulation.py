import numpy as np

def generateEquallySpacedTimes(maxtime, timecount):
    """Generate equally spaced times.
    
    >>> generateEquallySpacedTimes(1., 5)
    array([ 0.  ,  0.25,  0.5 ,  0.75,  1.  ])
    """
    return np.linspace(0., maxtime, timecount)

if __name__ == '__main__':
    import doctest
    doctest.testmod()
