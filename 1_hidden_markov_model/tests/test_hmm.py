
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
        data = np.array([[1,6]])
        k = 2
        max_iterations = 10
        threshold = 1e-5
        m = hmm.HMM(data, k, max_iterations, threshold)
        m.initialize()
        m.pi = np.array([.5,.5])
        m.A = np.array([[.5,.5],[.5,.5]])
        m.B = np.array([1, 5])
        m.forward()
        # check that the second class if more likely at the end
        final_class_probs = (m.alphas[:,-1:,:] / np.sum(m.alphas[:,-1:,:]))[0][0]
        self.assertTrue(final_class_probs[0] < final_class_probs[1])

        # multiple data case
        data = np.array([[1,6], [7,1]])
        k = 2
        max_iterations = 10
        threshold = 1e-5
        m = hmm.HMM(data, k, max_iterations, threshold)
        m.initialize()
        m.pi = np.array([.5,.5])
        m.A = np.array([[.5,.5],[.5,.5]])
        m.B = np.array([1, 5])
        m.forward()
        # check that the second class if more likely at the end
        final_class_probs = (m.alphas[:,-1:,:] / np.sum(m.alphas[:,-1:,:]))
        probs_1 = final_class_probs[0][0]
        probs_2 = final_class_probs[1][0]

        self.assertTrue(probs_1[0] < probs_1[1])
        self.assertTrue(probs_2[0] > probs_2[1])

    def test_backward(self):
        data = np.array([[1, 1, 1, 1]])
        k = 2
        max_iterations = 10
        threshold = 1e-5
        m = hmm.HMM(data, k, max_iterations, threshold)
        m.initialize()
        m.pi = np.array([.5,.5])
        m.A = np.array([[1,0],[0,1]])
        m.B = np.array([1, 10])
        m.backward()
        init_probs = m.betas[0][0]
        self.assertTrue(init_probs[0] > init_probs[1])

        data = np.array([[1,1,1,1], [5,5,5,5]])
        k = 2
        max_iterations = 10
        threshold = 1e-5
        m = hmm.HMM(data, k, max_iterations, threshold)
        m.initialize()
        m.pi = np.array([.5,.5])
        m.A = np.array([[1,0],[0,1]])
        m.B = np.array([1, 5])
        m.backward()
        init_probs_1 = m.betas[0][0]
        self.assertTrue(init_probs_1[0] > init_probs_1[1])
        init_probs_2 = m.betas[1][0]
        self.assertTrue(init_probs_2[0] < init_probs_2[1])

    def test_e_step(self):
        data = np.array([[1]])
        k = 2
        max_iterations = 10
        threshold = 1e-5
        m = hmm.HMM(data, k, max_iterations, threshold)
        m.initialize()
        m.pi = np.array([.5,.5])
        m.A = np.array([[.9,.1],[.1,.9]])
        m.B = np.array([1, 5])
        m.e_step()
        print m.gammas
        # what even is gamma?

    def test_m_step(self):
        data = np.array([[1,5], [5,1], [1,1]])
        k = 2
        max_iterations = 10
        threshold = 1e-5
        m = hmm.HMM(data, k, max_iterations, threshold)
        m.initialize()
        m.pi = np.array([.5,.5])
        m.A = np.array([[.5,.5],[.5,.5]])
        m.B = np.array([1, 5])
        for _ in range(100):
            m.e_step()
            m.m_step()
        print m.A
        print m.B
        print m.pi
        # what even is gamma?

    def test_hmm_on_generated_data(self):
        A = np.array([[1,0],[0,1]])
        B = np.array([[1],[10]])
        pi = np.array([.5,.5])
        T = 2
        dist = np.random.poisson
        num_samples = 10
        data = generate_data.generate_data(A, B, pi, T, dist, num_samples)
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




if __name__ == '__main__':
    unittest.main()