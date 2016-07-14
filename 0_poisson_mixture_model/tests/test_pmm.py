
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import unittest

path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(os.path.abspath(path))

import mm
import pmm
import utils

class TestPMM(unittest.TestCase):

    def test_m_step(self):
        data = np.array([[16],[28],[11]])
        means = np.array([[11],[28],[16]])
        phis = np.ones(3) / 3.
        density = utils.log_poisson
        w, log_prob = mm.e_step(data, means, phis, density)
        k = len(phis)
        new_means, new_phis = pmm.m_step(data, k, w)
        print 'original means: {}'.format(means)
        print 'new means: {}'.format(new_means)

    def test_e_step(self):
        data = np.array([[16],[28],[11]])
        means = np.array([[11],[28],[16]])
        phis = np.ones(3) / 3.
        density = utils.log_poisson
        w, log_prob = mm.e_step(data, means, phis, density)
        print data
        print means
        print w
        print log_prob


if __name__ == '__main__':
    unittest.main()