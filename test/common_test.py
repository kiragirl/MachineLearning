import unittest
import numpy as np

class MyTestCase(unittest.TestCase):
    def test_something(self):
        array_3d_random = np.random.randint(1, 10, size=(2, 3, 3))
        print("\n填充随机数值后的3维数组:\n", array_3d_random)
        print("\n填充随机数值后的3维数组:\n", array_3d_random[0])
        #self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
