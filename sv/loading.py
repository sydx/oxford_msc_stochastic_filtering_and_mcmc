from collections import OrderedDict

import numpy as np
from pandas import DataFrame

from sv import SVData, CorTiming

def loadSVDataFromBUGSDataset(filepath, logreturnforward, logreturnscale, dtfilepath=None):
    dts = None
    if dtfilepath is not None:
        with open(dtfilepath) as f:
            content = f.readlines()
            dts = np.array([float(x) for x in content[1:-1]])
    
    with open(filepath) as f:
        content = f.readlines()
    
    logreturns = np.array([float(x) for x in content[1:-1]])
    times = range(len(logreturns))
    
    if dts is not None:
        svdf = DataFrame(OrderedDict((('logreturn', logreturns), ('dt', dts))), index=times)
    else:
        svdf = DataFrame(OrderedDict((('logreturn', logreturns),)), index=times)
    
    return SVData(
            sourcekind='loader',
            source=loadSVDataFromBUGSDataset,
            svdf=svdf,
            params=None,
            cortiming=CorTiming.unknown,
            logreturnforward=logreturnforward,
            logreturnscale=logreturnscale)
