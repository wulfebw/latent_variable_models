
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import unittest

path = os.path.join(os.path.dirname(__file__), os.pardir, 'scripts')
sys.path.append(os.path.abspath(path))

import generate_data
import hmm

class TestHMM(unittest.TestCase):

    def test_forward(self):
        data = np.array([1,10])
        k = 2
        max_iterations = 10
        threshold = 1e-5
        m = hmm.HMM(data, k, max_iterations, threshold)
        m.initialize()
        m.pi = np.array([.5,.5])
        m.A = np.array([[.5,.5],[.5,.5]])
        m.B = np.array([1, 10])
        m.forward()
        # check that the second class is more likely at the end
        self.assertTrue(m.alphas[-1,0] < m.alphas[-1,1])

        # multiple data case
        data = np.array([1,6,7,1])
        k = 2
        max_iterations = 10
        threshold = 1e-5
        m = hmm.HMM(data, k, max_iterations, threshold)
        m.initialize()
        m.pi = np.array([.5,.5])
        m.A = np.array([[.5,.5],[.5,.5]])
        m.B = np.array([1, 5])
        m.forward()
        # check that the first class is more likely at the end
        self.assertTrue(m.alphas[-1,0] > m.alphas[-1,1])

    def test_backward(self):
        data = np.array([10, 1, 1, 1])
        k = 2
        max_iterations = 10
        threshold = 1e-5
        m = hmm.HMM(data, k, max_iterations, threshold)
        m.initialize()
        m.pi = np.array([.5,.5])
        m.A = np.array([[1,0],[0,1]])
        m.B = np.array([1, 10])
        m.backward()
        self.assertTrue(m.betas[0,0] > m.betas[0,1])

        data = np.array([1,1,1,1,5,5,5,5])
        k = 2
        max_iterations = 10
        threshold = 1e-5
        m = hmm.HMM(data, k, max_iterations, threshold)
        m.initialize()
        m.pi = np.array([.5,.5])
        m.A = np.array([[1,0],[0,1]])
        m.B = np.array([1, 5])
        m.backward()
        self.assertTrue(m.betas[0, 0] < m.betas[0, 1])

    def test_e_step(self):
        data = np.array([1,10,1,10,1,10])
        k = 2
        max_iterations = 10
        threshold = 1e-5
        m = hmm.HMM(data, k, max_iterations, threshold)
        m.initialize()
        m.pi = np.array([.5,.5])
        m.A = np.array([[.5,.5],[.5,.5]])
        m.B = np.array([1, 10])
        m.e_step()
        print m.gammas
        print m.etas
        # what even is gamma?

    def test_m_step(self):
        data = np.array([1,10,1,10,1,10,1,10])
        # data = np.array([1, 1])
        k = 2
        max_iterations = 10
        threshold = 1e-5
        m = hmm.HMM(data, k, max_iterations, threshold)
        m.initialize()
        m.pi = np.array([.5,.5])
        m.A = np.array([[.5,.5],[.5,.5]])
        m.B = np.array([1, 10])
        for _ in range(10):
            m.e_step()
            m.m_step()
        print m.A
        print m.B
        print m.pi
        # what even is gamma?

    def test_hmm_on_generated_data(self):
        # np.random.seed(2)

        # k = 2 case
        A = np.array([[0,1],[1,0]])
        B = np.array([[2],[10]])
        pi = np.array([.5,.5])
        T = 100
        dist = np.random.poisson
        data = generate_data.generate_data(A, B, pi, T, dist)
        print data

        k = 2
        max_iterations = 10
        threshold = 1e-5
        m = hmm.HMM(data, k, max_iterations, threshold)
        m.fit()
        print 'actual A: {}'.format(A)
        print 'learned A: {}'.format(m.A)
        print 'actual B: {}'.format(B)
        print 'learned B: {}'.format(m.B)

        # k = 3 case
        A = np.array([[.33,.33,.33],[.33,.33,.33],[.33,.33,.33]])
        B = np.array([[2],[10],[20]])
        pi = np.array([.5,.5,.5])
        T = 100
        dist = np.random.poisson
        data = generate_data.generate_data(A, B, pi, T, dist)
        print data

        k = 3
        max_iterations = 50
        threshold = 1e-5
        m = hmm.HMM(data, k, max_iterations, threshold)
        m.fit()
        print 'actual A: {}'.format(A)
        print 'learned A: {}'.format(m.A)
        print 'actual B: {}'.format(B)
        print 'learned B: {}'.format(m.B)

        # k = 4 case
        A = np.array([[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,0,0,0]])
        B = np.array([[2],[10],[20],[30]])
        pi = np.array([.25,.25,.25,.25])
        T = 50
        dist = np.random.poisson
        data = generate_data.generate_data(A, B, pi, T, dist)
        print data

        k = 4
        max_iterations = 50
        threshold = 1e-10
        m = hmm.HMM(data, k, max_iterations, threshold, verbose=True, seed=np.random.randint(100))
        m.fit()
        print 'actual A: {}'.format(A)
        print 'learned A: {}'.format(m.A)
        print 'actual B: {}'.format(B)
        print 'learned B: {}'.format(m.B)



if __name__ == '__main__':
    unittest.main()