import unittest

import utils.collections

class CollectionsTest(unittest.TestCase):
    
    def test_DiagonalArray(self):
        a = utils.collections.DiagonalArray(5)
        
        a[0,0] = 0
        a[1,0], a[1,1] = 10, 20
        a[2,0], a[2,1], a[2,2] = 30, 40, 50
        a[3,0], a[3,1], a[3,2], a[3,3] = 60, 70, 80, 90
        a[4,0], a[4,1], a[4,2], a[4,3], a[4,4] = 100, 110, 120, 130, 140
        
        self.assertEqual(len(a), 15)
        
        self.assertEqual(a[0,0], 0)
        self.assertEqual(a[1,0], 10)
        self.assertEqual(a[1,1], 20)
        self.assertEqual(a[2,0], 30)
        self.assertEqual(a[2,1], 40)
        self.assertEqual(a[2,2], 50)
        self.assertEqual(a[3,0], 60)
        self.assertEqual(a[3,1], 70)
        self.assertEqual(a[3,2], 80)
        self.assertEqual(a[3,3], 90)
        self.assertEqual(a[4,0], 100)
        self.assertEqual(a[4,1], 110)
        self.assertEqual(a[4,2], 120)
        self.assertEqual(a[4,3], 130)
        self.assertEqual(a[4,4], 140)
        
        self.assertEqual(a[0,0], 0)
        self.assertEqual(a[0,1], 10)
        self.assertEqual(a[1,1], 20)
        self.assertEqual(a[0,2], 30)
        self.assertEqual(a[1,2], 40)
        self.assertEqual(a[2,2], 50)
        self.assertEqual(a[0,3], 60)
        self.assertEqual(a[1,3], 70)
        self.assertEqual(a[2,3], 80)
        self.assertEqual(a[3,3], 90)
        self.assertEqual(a[0,4], 100)
        self.assertEqual(a[1,4], 110)
        self.assertEqual(a[2,4], 120)
        self.assertEqual(a[3,4], 130)
        self.assertEqual(a[4,4], 140)
        
        self.assertEqual(a._indextokey(0), (0, 0))
        self.assertEqual(a._indextokey(1), (1, 0))
        self.assertEqual(a._indextokey(2), (1, 1))
        self.assertEqual(a._indextokey(3), (2, 0))
        self.assertEqual(a._indextokey(4), (2, 1))
        self.assertEqual(a._indextokey(5), (2, 2))
        self.assertEqual(a._indextokey(6), (3, 0))
        self.assertEqual(a._indextokey(7), (3, 1))
        self.assertEqual(a._indextokey(8), (3, 2))
        self.assertEqual(a._indextokey(9), (3, 3))
        self.assertEqual(a._indextokey(10), (4, 0))
        self.assertEqual(a._indextokey(11), (4, 1))
        self.assertEqual(a._indextokey(12), (4, 2))
        self.assertEqual(a._indextokey(13), (4, 3))
        self.assertEqual(a._indextokey(14), (4, 4))

        values = []
        for v in a: values.append(v)
        self.assertSequenceEqual(tuple(a), (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140))
        
        keys = []
        for k in a.keys():
            keys.append(k)
        self.assertSequenceEqual(keys, ((0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2), (3, 3), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4)))

        keys, values = [], []
        for k, v in a.items():
            keys.append(k)
            values.append(v)
        self.assertSequenceEqual(keys, ((0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2), (3, 3), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4)))
        self.assertSequenceEqual(values, (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140))
        
        self.assertEqual(a.mindim(1), 1)
        self.assertEqual(a.mindim(2), 2)
        self.assertEqual(a.mindim(3), 2)
        self.assertEqual(a.mindim(4), 3)
        self.assertEqual(a.mindim(5), 3)
        self.assertEqual(a.mindim(6), 3)
        self.assertEqual(a.mindim(7), 4)
        self.assertEqual(a.mindim(8), 4)
        self.assertEqual(a.mindim(9), 4)
        self.assertEqual(a.mindim(10), 4)
        self.assertEqual(a.mindim(11), 5)
        self.assertEqual(a.mindim(12), 5)
        self.assertEqual(a.mindim(13), 5)
        self.assertEqual(a.mindim(14), 5)
        self.assertEqual(a.mindim(15), 5)

    def test_SubdiagonalArray(self):
        a = utils.collections.SubdiagonalArray(5)
        a[1,0] = 0
        a[2,0], a[2,1] = 10, 20
        a[3,0], a[3,1], a[3,2] = 30, 40, 50
        a[4,0], a[4,1], a[4,2], a[4,3] = 60, 70, 80, 90
        
        self.assertEqual(len(a), 10)
        
        self.assertEqual(a[1,0], 0)
        self.assertEqual(a[2,0], 10)
        self.assertEqual(a[2,1], 20)
        self.assertEqual(a[3,0], 30)
        self.assertEqual(a[3,1], 40)
        self.assertEqual(a[3,2], 50)
        self.assertEqual(a[4,0], 60)
        self.assertEqual(a[4,1], 70)
        self.assertEqual(a[4,2], 80)
        self.assertEqual(a[4,3], 90)
        
        self.assertEqual(a[0,1], 0)
        self.assertEqual(a[0,2], 10)
        self.assertEqual(a[1,2], 20)
        self.assertEqual(a[0,3], 30)
        self.assertEqual(a[1,3], 40)
        self.assertEqual(a[2,3], 50)
        self.assertEqual(a[0,4], 60)
        self.assertEqual(a[1,4], 70)
        self.assertEqual(a[2,4], 80)
        self.assertEqual(a[3,4], 90)
        
        self.assertEqual(a._indextokey(0), (1, 0))
        self.assertEqual(a._indextokey(1), (2, 0))
        self.assertEqual(a._indextokey(2), (2, 1))
        self.assertEqual(a._indextokey(3), (3, 0))
        self.assertEqual(a._indextokey(4), (3, 1))
        self.assertEqual(a._indextokey(5), (3, 2))
        self.assertEqual(a._indextokey(6), (4, 0))
        self.assertEqual(a._indextokey(7), (4, 1))
        self.assertEqual(a._indextokey(8), (4, 2))
        self.assertEqual(a._indextokey(9), (4, 3))
        
        values = []
        for v in a: values.append(v)
        self.assertSequenceEqual(values, (0, 10, 20, 30, 40, 50, 60, 70, 80, 90))

        keys = []
        for k in a.keys():
            keys.append(k)
        self.assertSequenceEqual(keys, ((1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (4, 2), (4, 3)))

        keys, values = [], []
        for k, v in a.items():
            keys.append(k)
            values.append(v)
        self.assertSequenceEqual(keys, ((1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (4, 2), (4, 3)))
        self.assertSequenceEqual(values, (0, 10, 20, 30, 40, 50, 60, 70, 80, 90))

        self.assertEqual(a.mindim(1), 2)
        self.assertEqual(a.mindim(2), 3)
        self.assertEqual(a.mindim(3), 3)
        self.assertEqual(a.mindim(4), 4)
        self.assertEqual(a.mindim(5), 4)
        self.assertEqual(a.mindim(6), 4)
        self.assertEqual(a.mindim(7), 5)
        self.assertEqual(a.mindim(8), 5)
        self.assertEqual(a.mindim(9), 5)
        self.assertEqual(a.mindim(10), 5)
        
if __name__ == '__main__':
    unittest.main()
    