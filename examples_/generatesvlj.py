import numpy as np
import matplotlib.pyplot as plt

import maths.numpyutils as npu
import sv.generation
import sv.visualisation

from svparams import params_pitt2014_fig1

randomstate = npu.randomstate()

timecount = 2001  # 0, 1, ..., 1000

generator = sv.generation.SVDataGenerator(
        timecount=timecount,
        params=params_pitt2014_fig1,
        cortiming=sv.CorTiming.coratsametime,
        logreturnforward=False,
        logreturnscale=100.,
        randomstate=randomstate,
        usestratonovichcorrection=False)
svdata = generator.generate()

print(svdata)

fig = plt.figure()
sv.visualisation.makesvdataplot(fig, svdata)
plt.show()
