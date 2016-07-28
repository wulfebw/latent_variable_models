
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import unittest

path = os.path.join(os.path.dirname(__file__), os.pardir, 'scripts')
sys.path.append(os.path.abspath(path))

import sbm
import utils

class TestSBM(unittest.TestCase):

    def test_initialize(self):
        model = sbm.SBM()
        k = 3
        data = np.array([[1,1,1],[1,1,1],[1,1,1]])
        model.initialize(data, k)
        print model.pis
        print model.gammas
        print model.taus

    def test_m_step(self):
        taus = np.array([[.5,.5],[.9,.1]])
        data = np.array([[1,2],[2,3]])
        k = 2
        model = sbm.SBM()
        model.initialize(data, k)
        model.taus = taus
        model.m_step()
        actual_pi = model.pis
        actual_gammas = model.gammas

        expected_pi = [.7, .3]
        expected_gammas = np.array([[2.2857,0],[0,0]])
        self.assertAlmostEquals(expected_pi[0], actual_pi[0])
        self.assertAlmostEquals(expected_pi[1], actual_pi[1])
        self.assertAlmostEquals(expected_gammas[0,0], actual_gammas[0,0], 3)

    def test_m_step_on_fake_data(self):
        # 3 nodes, 2 latent classes
        data = np.array([[0,1,1],[1,0,0],[1,0,0]])
        N = len(data)
        k = 2

        model = sbm.SBM()
        model.initialize(data, k)
        model.taus = np.array([[1,0],[0,1],[0,1.]])
        model.m_step()
        self.assertAlmostEquals(model.pis[0], .3333, 3)
        self.assertAlmostEquals(model.pis[1], .6667, 3)
        self.assertAlmostEquals(model.gammas[0,0], 0, 3)
        self.assertAlmostEquals(model.gammas[0,1], 1, 3)
        self.assertAlmostEquals(model.gammas[1,0], 1, 3)
        self.assertAlmostEquals(model.gammas[1,1], 0, 3)

    def test_e_step(self):
        np.random.seed(1)
        # 3 nodes, 2 latent classes
        data = np.array([[0,1,1],[1,0,10],[1,10,0]])
        N = len(data)
        k = 2

        model = sbm.SBM(e_iterations=1)
        model.initialize(data, k)
        model.taus = np.ones((N, k)) / k
        for _ in range(10):
            model.m_step()
            model.e_step()
        print model.taus

    def test_e_step_on_fake_data(self):
        np.random.seed(1)

        # k = 2 case
        pis = np.array([.7,.3])
        gammas = np.array([[15,1],[1,15]])
        num_nodes = 100
        data, ks = utils.generate_data(num_nodes, pis, gammas)
        k = len(pis)
        N = len(data)

        model = sbm.SBM(e_iterations=2)
        model.initialize(data, k)
        model.pis = pis
        model.gammas = gammas
        model.taus = np.ones((N, k)) / k
        model.e_step()

        actuals = np.argmax(model.taus, axis=1)
        common = len(ks[ks == actuals])
        result = common == len(ks) or common == 0
        self.assertTrue(result)

        # # k = 3 case
        # pis = np.array([.1,.4,.5])
        # gammas = np.array( [[1,  20, 100],
        #                     [20, 200, 300],
        #                     [100, 300, 500]])
        # num_nodes = 50
        # data, ks = utils.generate_data(num_nodes, pis, gammas)
        # k = len(pis)
        # N = len(data)

        # model = sbm.SBM(e_iterations=5)
        # model.initialize(data, k)
        # model.pis = pis
        # model.gammas = gammas
        # model.taus = np.ones((N, k)) / k
        # model.e_step()

        # actuals = np.argmax(model.taus, axis=1)
        # common = len(ks[ks == actuals])
        # result = common == len(ks) or common == 0

        # print ks
        # print actuals

        self.assertTrue(result)

    def test_fit_generated_data(self):
        # k = 2 case
        np.random.seed(20)
        pis = np.array([.5,.5])
        gammas = np.array([[15,1],[1,15]])

        print pis
        print gammas
        num_nodes = 50
        data, ks = utils.generate_data(num_nodes, pis, gammas)
        k = len(pis)

        model = sbm.SBM(e_iterations=25)
        model.fit(data, k, max_iterations=100, threshold=1e-3)
        print model.pis
        print model.gammas

        # k = 3 case
        np.random.seed(20)
        pis = np.array([.3,.2,.5])
        gammas = np.array([[40,20,30],
                           [20,50,70],
                           [30,70,90]])

        print pis
        print gammas
        num_nodes = 25
        data, ks = utils.generate_data(num_nodes, pis, gammas)
        k = len(pis)

        model = sbm.SBM(e_iterations=5)
        model.fit(data, k, max_iterations=50, threshold=1e-3)
        print model.pis
        print model.gammas

    def test_fit_real_data(self):
        np.random.seed(3)
        input_filepath = '../data/tree_net.csv'
        data, keys = utils.load_data(input_filepath)
        k = 4
        model = sbm.SBM(e_iterations=1)
        model.initialize(data, k)
        model.fit(data, k, max_iterations=100, threshold=1e-5)
        print model.pis
        print model.gammas

if __name__ == '__main__':
    unittest.main()