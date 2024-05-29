import unittest

import numpy as np
import pandas as pd

from filtering.generation import StateSpaceModelDataGenerator

class GenerationTest(unittest.TestCase):
    
    def test_StateSpaceModelDataGenerator(self):
        generator = StateSpaceModelDataGenerator(np.random.RandomState(seed=42))
        self.assertEqual(generator.addnoise('u', var=4., lag=1), 0)
        self.assertEqual(generator.addnoise('v', var=3., lag=0), 1)
        self.assertEqual(generator.addnoise('w', var=5., lag=3), 2)
        self.assertEqual(generator.addjump('J1', intensity=0.1, sd=100.), 0)
        self.assertEqual(generator.addjump('J2', intensity=0.8, sd=50000.), 1)
        def x(time, processname, data):
            assert processname == 'x'
            if time == 0: return 100.0
            return data.process('x', time - 1) + data.noise('u', time)
        def y(time, processname, data):
            assert processname == 'y'
            return 2.0 * data.process('x', time) + data.noise('v', time) + data.noise('w', time)
        self.assertEqual(generator.addprocess('x', x), 0)
        self.assertEqual(generator.addprocess('y', y), 1)
        generator.setnoisecors((-.25, -.5, .3))
        generator.settimecount(5)
        data1 = generator.generate()
        data2 = generator.generate()
        
    def test_StateSpaceModelDataGenerator_SVLJ(self):
        meanlogvar = -0.1706
        persistence = 0.9755
        cor = -0.2699
        voloflogvar = 0.1464
        jumpintensity = 0.25
        jumpvol = 10.
        
        logreturnscale = 100.
        initialprice = 1000.
        
        constterm = meanlogvar * (1. - persistence) 
        
        generator = StateSpaceModelDataGenerator(np.random.RandomState(seed=42))
        self.assertEqual(generator.addnoise('epsilon', var=1., lag=0), 0)
        self.assertEqual(generator.addnoise('eta', var=1., lag=1), 1)
        generator.setnoisecors((cor,))
        self.assertEqual(generator.addjump('jump', intensity=jumpintensity, sd=jumpvol), 0)
        def logvar(time, processname, data):
            if time == 0: return meanlogvar
            return constterm + \
                    persistence * data.process('logvar', time - 1) + \
                    voloflogvar * data.noise('eta', time)
        def logreturn(time, processname, data):
            if time == 0: return np.nan
            return data.noise('epsilon', time) * \
                    np.exp(.5 * data.process('logvar', time)) + \
                    data.jump('jump', time)
        def logpricefrombackwardlogreturn(time, processname, data):
            prevlogprice = np.log(initialprice) if time == 0 else data.process('logprice', time - 1)
            logreturn = data.process('logreturn', time)
            return prevlogprice if np.isnan(logreturn) else prevlogprice + logreturn / logreturnscale
        def price(time, processname, data):
            return np.exp(data.process('logprice', time)) 
        self.assertEqual(generator.addprocess('logvar', logvar), 0)
        self.assertEqual(generator.addprocess('logreturn', logreturn), 1)
        self.assertEqual(generator.addprocess('logprice', logpricefrombackwardlogreturn), 2)
        self.assertEqual(generator.addprocess('price', price), 3)
        generator.settimecount(10)
        data = generator.generate()
        processdf = data.processdf()
        jumpflagdf = data.jumpflagdf(colname=lambda idx, name: '%sflag' % name)
        jumpdf = data.jumpdf()
        df = pd.concat((processdf, jumpflagdf, jumpdf), axis=1)
        print(df)
        #import pylab
        #pylab.plot(data.processdf())
        #pylab.show()
        
if __name__ == '__main__':
    unittest.main()
    