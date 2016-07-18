
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import unittest

path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(os.path.abspath(path))

import utils

class TestUtils(unittest.TestCase):

    def test_poisson(self):
        mean = 26
        num_samples = 20
        samples = np.empty(num_samples)
        for idx in range(num_samples):
            samples[idx] = utils.poisson(idx, mean)
        plt.scatter(range(num_samples), samples)
        plt.show()

    def test_log_poisson(self):
        mean = 26
        num_samples = 20
        samples = np.empty(num_samples)
        for idx in range(num_samples):
            samples[idx] = np.exp(utils.log_poisson(idx, mean))
        plt.scatter(range(num_samples), samples)
        plt.show()

    def test_log_factorial(self):
        x = 20 
        actual = utils.log_factorial(x)
        expected = np.log(np.math.factorial(x))
        self.assertEquals(expected, actual)


        

if __name__ == '__main__':
    unittest.main()

