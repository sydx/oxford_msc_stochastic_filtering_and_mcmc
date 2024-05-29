import numpy as np

import sv

# The parameters used in Fig. 1 in (Pitt et al., 2014):
params_pitt2014_fig1 = sv.Params(
        meanlogvar    = 0.25,
        persistence   = 0.975,
        cor           = -0.8,
        voloflogvar   = np.sqrt(0.025),
        jumpintensity = 0.01,
        jumpvol       = 10.)

# SVL, Dataset 1
params_svl_ds1 = sv.Params(
        meanlogvar    = -0.6809,
        persistence   = 0.9809,
        cor           = -0.04357,
        voloflogvar   = 0.1549,
        jumpintensity = 0.,
        jumpvol       = 1.)

# SVL, Dataset 2
params_svl_ds2 = sv.Params(
        meanlogvar    = -0.2213,
        persistence   = 0.9696,
        cor           = -0.2828,
        voloflogvar   = 0.1633,
        jumpintensity = 0.,
        jumpvol       = 1.)

# SVL, Dataset 3
params_svl_ds3 = sv.Params(
        meanlogvar    = -0.7431,
        persistence   = 0.935,
        cor           = -0.3497,
        voloflogvar   = 0.2664,
        jumpintensity = 0.,
        jumpvol       = 1.)

# SVL, Dataset 4
params_svl_ds4 = sv.Params(
        meanlogvar    = 0.1105,
        persistence   = 0.9707,
        cor           = -0.7859,
        voloflogvar   = 0.1746,
        jumpintensity = 0.,
        jumpvol       = 1.)

# SVL, Dataset 5
params_svl_ds5 = sv.Params(
        meanlogvar    = -0.0464,
        persistence   = 0.9844,
        cor           = -0.7364,
        voloflogvar   = 0.1762,
        jumpintensity = 0.,
        jumpvol       = 1.)

# SVL, Dataset 6
params_svl_ds6 = sv.Params(
        meanlogvar    = -0.8146,
        persistence   = 0.9162,
        cor           = -0.852,
        voloflogvar   = 0.3655,
        jumpintensity = 0.,
        jumpvol       = 1.)

# SVL, Dataset 7
params_svl_ds7 = sv.Params(
        meanlogvar    = -0.6046,
        persistence   = 0.9456,
        cor           = -0.7251,
        voloflogvar   = 0.276,
        jumpintensity = 0.,
        jumpvol       = 1.)

# SVL, Dataset 8
params_svl_ds8 = sv.Params(
        meanlogvar    = 0.1942,
        persistence   = 0.9432,
        cor           = -0.7385,
        voloflogvar   = 0.2623,
        jumpintensity = 0.,
        jumpvol       = 1.)

# SVL, Dataset 9
params_svl_ds9 = sv.Params(
        meanlogvar    = 0.08395,
        persistence   = 0.9324,
        cor           = -0.7319,
        voloflogvar   = 0.2908,
        jumpintensity = 0.,
        jumpvol       = 1.)

# SVL, Dataset 10
params_svl_ds10 = sv.Params(
        meanlogvar    = 0.1025,
        persistence   = 0.9507,
        cor           = -0.6639,
        voloflogvar   = 0.239,
        jumpintensity = 0.,
        jumpvol       = 1.)

# SVL, Dataset 11
params_svl_ds11 = sv.Params(
        meanlogvar    = 0.3487,
        persistence   = 0.8932,
        cor           = -0.5037,
        voloflogvar   = 0.4168,
        jumpintensity = 0.,
        jumpvol       = 1.)

# SVL, Dataset 12
params_svl_ds12 = sv.Params(
        meanlogvar    = 0.5455,
        persistence   = 0.9842,
        cor           = -0.129,
        voloflogvar   = 0.1665,
        jumpintensity = 0.,
        jumpvol       = 1.)

# SVL, Dataset 13
params_svl_ds13 = sv.Params(
        meanlogvar    = 0.1241,
        persistence   = 0.9632,
        cor           = -0.3272,
        voloflogvar   = 0.1705,
        jumpintensity = 0.,
        jumpvol       = 1.)

# SVL2, Dataset 1
params_svl2_ds1 = sv.Params(
        meanlogvar    = -0.6265,
        persistence   = 0.9835,
        cor           = -0.1452,
        voloflogvar   = 0.1535,
        jumpintensity = 0.,
        jumpvol       = 1.)

# SVL2, Dataset 2
params_svl2_ds2 = sv.Params(
        meanlogvar    = -0.2067,
        persistence   = 0.9746,
        cor           = -0.2757,
        voloflogvar   = 0.1488,
        jumpintensity = 0.,
        jumpvol       = 1.)

# SVL2, Dataset 3
params_svl2_ds3 = sv.Params(
        meanlogvar    = -0.5395,
        persistence   = 0.966,
        cor           = -0.3446,
        voloflogvar   = 0.1864,
        jumpintensity = 0.,
        jumpvol       = 1.)

# SVL2, Dataset 4
params_svl2_ds4 = sv.Params(
        meanlogvar    = 0.06216,
        persistence   = 0.9802,
        cor           = -0.7922,
        voloflogvar   = 0.1522,
        jumpintensity = 0.,
        jumpvol       = 1.)

# SVL2, Dataset 5
params_svl2_ds5 = sv.Params(
        meanlogvar    = -0.151,
        persistence   = 0.9861,
        cor           = -0.8415,
        voloflogvar   = 0.1892,
        jumpintensity = 0.,
        jumpvol       = 1.)

# SVL2, Dataset 6
params_svl2_ds6 = sv.Params(
        meanlogvar    = -0.8758,
        persistence   = 0.958,
        cor           = -0.8292,
        voloflogvar   = 0.2721,
        jumpintensity = 0.,
        jumpvol       = 1.)

# SVL2, Dataset 7
params_svl2_ds7 = sv.Params(
        meanlogvar    = -0.6743,
        persistence   = 0.9715,
        cor           = -0.7427,
        voloflogvar   = 0.2223,
        jumpintensity = 0.,
        jumpvol       = 1.)

# SVL2, Dataset 8
params_svl2_ds8 = sv.Params(
        meanlogvar    = 0.1682,
        persistence   = 0.9735,
        cor           = -0.8042,
        voloflogvar   = 0.1832,
        jumpintensity = 0.,
        jumpvol       = 1.)

# SVL2, Dataset 9
params_svl2_ds9 = sv.Params(
        meanlogvar    = 0.06336,
        persistence   = 0.9697,
        cor           = -0.7909,
        voloflogvar   = 0.2003,
        jumpintensity = 0.,
        jumpvol       = 1.)

# SVL2, Dataset 10
params_svl2_ds10 = sv.Params(
        meanlogvar    = 0.08121,
        persistence   = 0.969,
        cor           = -0.7233,
        voloflogvar   = 0.1954,
        jumpintensity = 0.,
        jumpvol       = 1.)

# SVL2, Dataset 11
params_svl2_ds11 = sv.Params(
        meanlogvar    = 0.3195,
        persistence   = 0.9321,
        cor           = -0.5191,
        voloflogvar   = 0.3355,
        jumpintensity = 0.,
        jumpvol       = 1.)

# SVL2, Dataset 12
params_svl2_ds12 = sv.Params(
        meanlogvar    = 0.5619,
        persistence   = 0.9826,
        cor           = -0.2667,
        voloflogvar   = 0.1777,
        jumpintensity = 0.,
        jumpvol       = 1.)

# SVL2, Dataset 13
params_svl2_ds13 = sv.Params(
        meanlogvar    = 0.1183,
        persistence   = 0.9691,
        cor           = -0.3512,
        voloflogvar   = 0.1557,
        jumpintensity = 0.,
        jumpvol       = 1.)

params = {}
params[('svl' , 'dataset-1')] = params_svl_ds1
params[('svl' , 'dataset-2')] = params_svl_ds2
params[('svl' , 'dataset-3')] = params_svl_ds3
params[('svl' , 'dataset-4')] = params_svl_ds4
params[('svl' , 'dataset-5')] = params_svl_ds5
params[('svl' , 'dataset-6')] = params_svl_ds6
params[('svl' , 'dataset-7')] = params_svl_ds7
params[('svl' , 'dataset-8')] = params_svl_ds8
params[('svl' , 'dataset-9')] = params_svl_ds9
params[('svl' , 'dataset-10')] = params_svl_ds10
params[('svl' , 'dataset-11')] = params_svl_ds11
params[('svl' , 'dataset-12')] = params_svl_ds12
params[('svl' , 'dataset-13')] = params_svl_ds13
params[('svl2', 'dataset-1')] = params_svl2_ds1
params[('svl2', 'dataset-2')] = params_svl2_ds2
params[('svl2', 'dataset-3')] = params_svl2_ds3
params[('svl2', 'dataset-4')] = params_svl2_ds4
params[('svl2', 'dataset-5')] = params_svl2_ds5
params[('svl2', 'dataset-6')] = params_svl2_ds6
params[('svl2', 'dataset-7')] = params_svl2_ds7
params[('svl2', 'dataset-8')] = params_svl2_ds8
params[('svl2', 'dataset-9')] = params_svl2_ds9
params[('svl2', 'dataset-10')] = params_svl2_ds10
params[('svl2', 'dataset-11')] = params_svl2_ds11
params[('svl2', 'dataset-12')] = params_svl2_ds12
params[('svl2', 'dataset-13')] = params_svl2_ds13
