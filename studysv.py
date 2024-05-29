import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt

import maths.numpyutils as npu
import filtering.kalman
import filtering.particle
import filtering.run
import filtering.visualisation
import sv.filtering.gaussian
import sv.filtering.particle
import sv.filtering.unscented
import sv.generation
import sv.loading
import sv.visualisation

def runsvljparticlefilter(svdata, params, randomstate):
    initialdistribution = sv.generation.LogVarInitialDistribution(params, randomstate)
    transitiondistribution = sv.filtering.particle.SVLJLogVarTransitionDistribution(params, randomstate)
    weightingfunction = sv.filtering.particle.SVLJWeightingFunction(params)    
    particlecount = 1000
    predictedobservationsampler = sv.filtering.particle.SVLJPredictedObservationSampler(params, randomstate)
    stochfilter = filtering.particle.RegularisedResamplingParticleFilter(
            initialdistribution=initialdistribution,
            transitiondistribution=transitiondistribution,
            weightingfunction=weightingfunction,
            particlecount=particlecount,
            statedim=1,
            observationdim=1,
            randomstate=randomstate,
            predictedobservationsampler=predictedobservationsampler)
    return filtering.run.runfilter(svdata.svdf, params, stochfilter, 'logreturn', 'logvar')
    
def runsvl2particlefilter(svdata, params, randomstate):
    initialdistribution = sv.generation.LogVarInitialDistribution(params, randomstate)
    transitiondistribution = sv.filtering.particle.SVL2LogVarTransitionDistribution(params, randomstate)
    weightingfunction = sv.filtering.particle.SVL2WeightingFunction(params)    
    particlecount = 1000
    predictedobservationsampler = sv.filtering.particle.SVL2PredictedObservationSampler(params, randomstate)
    stochfilter = filtering.particle.RegularisedResamplingParticleFilter(
            initialdistribution=initialdistribution,
            transitiondistribution=transitiondistribution,
            weightingfunction=weightingfunction,
            particlecount=particlecount,
            statedim=1,
            observationdim=1,
            randomstate=randomstate,
            predictedobservationsampler=predictedobservationsampler)
    return filtering.run.runfilter(svdata.svdf, params, stochfilter, 'logreturn', 'logvar')
    
def runwcsvlparticlefilter(svdata, params, randomstate):
    initialdistribution = sv.generation.LogVarInitialDistribution(params, randomstate)
    transitiondistribution = sv.filtering.particle.WCSVLLogVarTransitionDistribution(params, randomstate)
    weightingfunction = sv.filtering.particle.WCSVLWeightingFunction(params)    
    particlecount = 1000
    predictedobservationsampler = sv.filtering.particle.WCSVLPredictedObservationSampler(params, randomstate)
    stochfilter = filtering.particle.MultinomialResamplingParticleFilter(
            initialdistribution=initialdistribution,
            transitiondistribution=transitiondistribution,
            weightingfunction=weightingfunction,
            particlecount=particlecount,
            statedim=1,
            observationdim=1,
            randomstate=randomstate,
            predictedobservationsampler=predictedobservationsampler)
    return filtering.run.runfilter(svdata.svdf, params, stochfilter, 'logreturn', 'logvar', dtcolumnname='dt')
    
def runsvlgaussianfilter(svdata, params, *args):
    stochfilter = sv.filtering.gaussian.SVLGaussianFilter(params.meanlogvar, params.logvaruncondvar(), params)
    return filtering.run.runfilter(svdata.svdf, params, stochfilter, 'logreturn', 'logvar')

def runsvl2gaussianfilter(svdata, params, *args):
    stochfilter = sv.filtering.gaussian.SVL2GaussianFilter(params.meanlogvar, params.logvaruncondvar(), params)
    return filtering.run.runfilter(svdata.svdf, params, stochfilter, 'logreturn', 'logvar')

def rununscentedkalmanfilter(svdata, params, *args):
    f = lambda x, w: params.meanlogvar * (1. - params.persistence) + params.persistence * x + params.voloflogvar * w
    h = lambda x, v: v * np.exp(.5*x)
    stochfilter = sv.filtering.unscented.UnscentedKalmanFilter(params.meanlogvar, params.logvaruncondvar(), 1., 1., params.cor, f, h)
    return filtering.run.runfilter(svdata.svdf, params, stochfilter, 'logreturn', 'logvar')

def runkalmanfilter(svdata, params, *args):
    mustar = .7979 * params.cor * params.voloflogvar
    gammastar = 1.1061 * params.cor * params.voloflogvar
    omega = -1.27
    varofxi = .5 * np.pi * np.pi
    
    x0 = params.meanlogvar
    P0 = params.logvaruncondvar()
    Q = varofxi - mustar*mustar - gammastar*gammastar / varofxi
    R = varofxi
        
    H = 1.
    b = omega

    sign = 1.
    observation = .1
    F = params.persistence - gammastar * sign / varofxi
    a = sign * (mustar + gammastar / varofxi * (np.log(observation * observation) - omega))
    
    stochfilter = filtering.kalman.KalmanFilter(x0, P0, Q, R, F, H, a, b)
    
    def observationtransform(observation, stochfilter):
        sign = -1. if observation < 0. else 1.
        stochfilter.F = params.persistence - gammastar * sign / varofxi
        stochfilter.a = sign * (mustar + gammastar / varofxi * (np.log(observation * observation) - omega))
        return np.log(observation * observation)
    
    return filtering.run.runfilter(svdata.svdf, params, stochfilter, 'logreturn', 'logvar', observationtransform=observationtransform)

def enrichsvdata(svdata, initialprice):
    if 'logprice' not in svdata.svdf.columns:
        svdata.svdf['logprice'] = sv.logreturntologprice(svdata.svdf, initialprice, svdata.logreturnforward, svdata.logreturnscale)
    if 'price' not in svdata.svdf.columns:
        svdata.svdf['price'] = np.exp(svdata.svdf['logprice'].values)

def examinesvdata(svdata):
    print(svdata)
    fig = plt.figure()
    sv.visualisation.makesvdataplot(fig, svdata)
    
def analyseparamsneighbourhood(svdata, params, includejumps, randomstate):
    parameterndarray = transformparameterndarray(np.array(params), includejumps)
    offsets = np.linspace(-.5, .5, 10)
    for dimension in range(params.dimensioncount):
        xs, ys = [], []
        parametername = params.getdimensionname(dimension)
        print('Perturbing %s...' % parametername)
        for offset in offsets:
            newparameterndarray = np.copy(parameterndarray)
            newparameterndarray[dimension] += offset
            xs.append(inversetransformparameterndarray(newparameterndarray, includejumps)[dimension])
            y = runsvljparticlefilter(svdata, sv.Params(*inversetransformparameterndarray(newparameterndarray, includejumps)), randomstate).stochfilter.loglikelihood
            ys.append(y)
        fig = plt.figure()
        plot = fig.add_subplot(111)
        plot.plot(xs, ys)
        plot.axvline(x=inversetransformparameterndarray(parameterndarray, includejumps)[dimension], color='red')
        plot.set_xlabel(parametername)
        plot.set_ylabel('loglikelihood')
        plt.show()
        
def transformparameterndarray(parameterndarray, includejumps):
    parameterndarray = npu.tondim1(parameterndarray)
    res = [
            parameterndarray[0],  # meanlogvar
            2. * np.arctanh(parameterndarray[1]), # persistence
            np.log(parameterndarray[2] * parameterndarray[2]), # voloflogvar
            2. * np.arctanh(parameterndarray[3]) # cor
        ]
    if includejumps:
        res.append(np.arctanh(2*parameterndarray[4] - 1)) # jumpintensity
        res.append(np.log(parameterndarray[5] * parameterndarray[5])) # jumpvol
    return np.array(res)
    
def inversetransformparameterndarray(parameterndarray, includejumps):
    parameterndarray = npu.tondim1(parameterndarray)
    res = [
            parameterndarray[0],  # meanlogvar
            np.tanh(.5 * parameterndarray[1]), # persistence
            np.sqrt(np.exp(parameterndarray[2])), # voloflogvar
            np.tanh(.5 * parameterndarray[3]) # cor
        ]
    if includejumps:
        res.append(.5 * (np.tanh(parameterndarray[4]) + 1)) # jumpintensity
        res.append(np.sqrt(np.exp(parameterndarray[5]))) # jumpvol
    else:
        res.append(0.)
        res.append(1.)
    return np.array(res)
        
def optimiseparams(svdata, initialguessparams, trueparams, filterrunner, includejumps, randomstate):
    def objectivefunction(transformedparameterndarray):
        mockinfinity = 10000.
        print(transformedparameterndarray)
        parameterndarray = inversetransformparameterndarray(transformedparameterndarray, includejumps)
        params = sv.Params(*parameterndarray)
        print(params)
        if params.persistence >= 0.99 or params.persistence <= -0.99:
            print('Parameter out of bound: persistence = %f\n' % params.persistence)
            return mockinfinity
        if params.cor <= -0.99 or params.cor >= 0.99:
            print('Parameter out of bound: cor = %f\n' % params.cor)
            return mockinfinity
        if params.voloflogvar <= 0.01:
            print('Parameter out of bound: voloflogvar = %f\n' % params.vologlogvar)
            return mockinfinity
        if params.jumpintensity < 0. or params.jumpintensity > 1.:
            print('Parameter out of bound: jumpintensity = %f\n' % params.jumpintensity)
            return mockinfinity
        if params.jumpvol < 0.:
            print('Parameter out of bound: jumpvol = %f\n' % params.jumpvol)
            return mockinfinity
        loglikelihood = filterrunner(svdata, params, randomstate).stochfilter.loglikelihood
        loglikelihood = np.asscalar(loglikelihood)
        print('Loglikelihood: %f\n' % loglikelihood)
        return -loglikelihood
    
    print('True parameters:')
    print(trueparams)
    trueparamsloglikelihood = -objectivefunction(transformparameterndarray(np.array(trueparams), includejumps))
    print('True params loglikelihood: %f\n' % trueparamsloglikelihood)
    
    print('Initial guess parameters:')
    print(initialguessparams)

    print('Running the optimisation routine (BFGS)...')
    res = opt.fmin_bfgs(
            objectivefunction,
            x0=transformparameterndarray(np.array(initialguessparams), includejumps),
            epsilon=0.1111,
            disp=True
            )
    
    res = sv.Params(*inversetransformparameterndarray(res, includejumps))
    
    print('Result:')
    print(res)
    
    return res
        
def generatesvdata(params, timecount, randomstate):
    generator = sv.generation.SVDataGenerator(
            timecount=timecount,
            params=params,
            cortiming=sv.CorTiming.coratsametime,
            logreturnforward=False,
            logreturnscale=100.,
            randomstate=randomstate,
            usestratonovichcorrection=False)
    return generator.generate()

def main():
    np.seterr(divide='raise', invalid='raise')    

    randomstate = npu.randomstate()

    params = sv.Params(
            meanlogvar=.65762,
            persistence=.96125,
            voloflogvar=np.sqrt(0.020053), # 0.1416
            cor=-.19,
            #jumpintensity=0.01,
            jumpintensity=0.,
            jumpvol=0.01)
    
    #params = sv.Params(meanlogvar=-0.2076, persistence=0.9745, cor=0.0, voloflogvar=0.0492, jumpintensity=0., jumpvol=1.)
    
    # SVL, ds1 -924.077823959 (600 particles)
    # params = sv.Params(meanlogvar=-0.5486, persistence=0.9861, cor=-0.1969, voloflogvar=0.149, jumpintensity=0., jumpvol=1.)
    # SVL, ds2 -2721.36910265 (600 particles)
    # params = sv.Params(meanlogvar=-0.1706, persistence=0.9755, cor=-0.2699, voloflogvar=0.1464, jumpintensity=0., jumpvol=1.)
    # SVL2, ds1 -923.050833581 (1000 particles)
    # params = sv.Params(meanlogvar=-0.5883, persistence=0.9853, cor=-0.1472, voloflogvar=0.1456, jumpintensity=0., jumpvol=1.)
    # SVL2, ds2 -2723.29157267 (1000 particles)
    # params = sv.Params(meanlogvar=-0.2076, persistence=0.9745, cor=-0.275, voloflogvar=0.1492 * 1.25, jumpintensity=0., jumpvol=1.)

    """    
    params = sv.Params(
        meanlogvar    = 0.25,
        persistence   = 0.975,
        #cor           = -0.8,
        cor = -0.5,
        voloflogvar   = np.sqrt(0.025),
        jumpintensity = 0.0,
        jumpvol       = 10.)
    """
    
    """
    params = sv.Params(
        meanlogvar    = 0.2048,
        persistence   = 0.6726,
        #cor           = -0.8,
        cor = 0.004101,
        voloflogvar   = 17.62,
        jumpintensity = 0.0,
        jumpvol       = 10.)

    """    
    params = sv.Params(
        meanlogvar    = -3.971,
        persistence   = 0.2338,
        #cor           = -0.8,
        cor = -0.9178,
        voloflogvar   = 0.01468,
        jumpintensity = 0.0,
        jumpvol       = 10.)

    # Wrong!
    # params = sv.Params(meanlogvar=-5.2076, persistence=0.9745, cor=0.275, voloflogvar=1.1492, jumpintensity=0., jumpvol=1.)
    # params = sv.Params(meanlogvar=-0.2076, persistence=0.9745, cor=-0.275, voloflogvar=0.1492, jumpintensity=0.01, jumpvol=10.)

    initialprice = 100.
    
    timecount = 2001  # 0, 1, ..., 1000

    print('Generating SV data...')
    svdata = generatesvdata(params, timecount, randomstate)
    
    print('Loading SV data...')
    # filepath = r"S:\dev\bodleian\dissertations\2016_Bilokon_Oxford_MSc_Bayesian-Methods-for-Solving-Estimation-and-Forecasting-Problems-in-the-High-Frequency-Trading-Environment\code\winbugs\datasets\dataset-1_GBPUSD_1981-10-01_1985-06-28.txt"
    # filepath = r"S:\dev\bodleian\dissertations\2016_Bilokon_Oxford_MSc_Bayesian-Methods-for-Solving-Estimation-and-Forecasting-Problems-in-the-High-Frequency-Trading-Environment\code\datasets\dataset-2_y.txt"
    filepath = r"S:\dev\bodleian\dissertations\2016_Bilokon_Oxford_MSc_Bayesian-Methods-for-Solving-Estimation-and-Forecasting-Problems-in-the-High-Frequency-Trading-Environment\code\datasets\dataset-14-ESM16\dataset-14-ESM16_y.txt"
    dtfilepath = r"S:\dev\bodleian\dissertations\2016_Bilokon_Oxford_MSc_Bayesian-Methods-for-Solving-Estimation-and-Forecasting-Problems-in-the-High-Frequency-Trading-Environment\code\datasets\dataset-14-ESM16\dataset-14-ESM16_dt.txt"
    svdata = sv.loading.loadSVDataFromBUGSDataset(filepath, logreturnforward=True, logreturnscale=100., dtfilepath=dtfilepath)

    enrichsvdata(svdata, initialprice)
        
    # action = 'examinesvdata'
    # action = 'analyseparamsneighbourhood'
    # action = 'optimiseparams'
    action = 'runsvljparticlefilteronceandanalyse'
    # action = 'runsvl2particlefilteronceandanalyse'
    # action = 'runwcsvlparticlefilteronceandanalyse'
    # action = 'runsvlgaussianfilteronceandanalyse'
    # action = 'runsvl2gaussianfilteronceandanalyse'
    # action = 'rununscentedkalmanfilterandanalyse'
    # action = 'runkalmanfilterandanalyse'

    print('Analysing SV data...')
    examinesvdata(svdata)
    
    print('Running action: %s...' % action)
    if action == 'examinesvdata':
        pass
    elif action == 'analyseparamsneighbourhood':
        analyseparamsneighbourhood(svdata, params, includejumps=False, randomstate=randomstate)
    elif action == 'optimiseparams':
        initialguessparams = sv.Params(
                meanlogvar=-.1,
                persistence=.975,
                voloflogvar=np.sqrt(.02),
                cor=-.8,
                jumpintensity=0.,
                jumpvol=1.)
        optimiseparams(svdata, initialguessparams, params, runsvljparticlefilter, includejumps=False, randomstate=randomstate)
    else:
        if action == 'runsvljparticlefilteronceandanalyse':
            filterrundata = runsvljparticlefilter(svdata, params, randomstate)
            fig = plt.figure()
            filtering.visualisation.makeparticlehistogram(fig, filterrundata.stochfilter)
        elif action == 'runsvl2particlefilteronceandanalyse':
            filterrundata = runsvl2particlefilter(svdata, params, randomstate)
            fig = plt.figure()
            filtering.visualisation.makeparticlehistogram(fig, filterrundata.stochfilter)
        elif action == 'runwcsvlparticlefilteronceandanalyse':
            filterrundata = runwcsvlparticlefilter(svdata, params, randomstate)
            fig = plt.figure()
            filtering.visualisation.makeparticlehistogram(fig, filterrundata.stochfilter)
        elif action == 'runsvlgaussianfilteronceandanalyse':
            filterrundata = runsvlgaussianfilter(svdata, params)
        elif action == 'runsvl2gaussianfilteronceandanalyse':
            filterrundata = runsvl2gaussianfilter(svdata, params)
        elif action == 'rununscentedkalmanfilterandanalyse':
            filterrundata = rununscentedkalmanfilter(svdata, params)
        elif action == 'runkalmanfilterandanalyse':
            filterrundata = runkalmanfilter(svdata, params)
        else:
            raise RuntimeError('Invalid action')
        print(filterrundata)    
        fig = plt.figure()
        filtering.visualisation.makefilterrunplot(fig, filterrundata.filterrundf)        
    
    plt.show()

if __name__ == '__main__':
    main()
