import unittest

import maths.stats
import utils.collections

class StatsTest(unittest.TestCase):
    def test_cor2cov(self):
        cors = utils.collections.SubdiagonalArray.create((-.25, -.5, .3))
        vars = (4., 3., 5.)
        covs = maths.stats.cor2cov(cors, vars)
        print(cors)
        print(covs)
            
if __name__ == '__main__':
    unittest.main()
    