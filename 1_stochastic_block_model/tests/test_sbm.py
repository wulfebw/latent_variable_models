
import copy
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
        model = sbm.SBM(e_iterations=100)
        model.pis = np.array([1/3.,2/3.])
        model.gammas = np.array([[1,10],
                                 [10,1]])
        model.taus = np.array([[.6,.4],[.4,.6],[.5,.5]])
        model.data = np.array([ [0,10,10],
                                [10,0,1],
                                [10,1,0]])
        model.N = len(model.data)
        model.k = len(model.pis)
        model.e_step()
        print model.taus

    def test_e_step_on_fake_data(self):
        
        # k = 2 case
        np.random.seed(1)
        pis = np.array([.7,.3])
        gammas = np.array([[15,1],[1,15]])
        num_nodes = 50
        data, ks = utils.generate_data(num_nodes, pis, gammas)
        k = len(pis)
        N = len(data)

        model = sbm.SBM(e_iterations=5)
        model.initialize(data, k)
        model.pis = pis
        model.gammas = gammas
        model.taus = np.ones((N, k)) / k
        model.e_step()

        actuals = np.argmax(model.taus, axis=1)
        common = len(ks[ks == actuals])
        result = common == len(ks) or common == 0
        self.assertTrue(result)

        # k = 3 case
        for s in range(4):
            np.random.seed(s)
            pis = np.array([.1,.4,.5])
            gammas = np.array( [[1,  10, 20],
                                [10, 30, 40],
                                [20, 40, 50]])
            num_nodes = 20
            data, ks = utils.generate_data(num_nodes, pis, gammas)
            k = len(pis)
            N = len(data)

            model = sbm.SBM(e_iterations=10)
            model.initialize(data, k)
            model.pis = pis
            model.gammas = gammas
            model.taus = np.ones((N, k)) / k
            model.e_step()

            actuals = np.argmax(model.taus, axis=1)
            common = len(ks[ks == actuals])
            result = common == len(ks) or common == 0

            print ks
            print actuals
            print s
            self.assertTrue(result)

        # k = 4 case
        for s in range(4):
            np.random.seed(s)
            pis = np.array([.1,.3,.5,.1])
            gammas = np.array( [[1,  10, 20, 30],
                                [10, 40, 50, 60],
                                [20, 50, 70, 80],
                                [30, 60, 80, 90]])
            num_nodes = 20
            data, ks = utils.generate_data(num_nodes, pis, gammas)
            k = len(pis)
            N = len(data)

            model = sbm.SBM(e_iterations=10)
            model.initialize(data, k)
            model.pis = pis
            model.gammas = gammas
            model.taus = np.ones((N, k)) / k
            model.e_step()

            actuals = np.argmax(model.taus, axis=1)
            common = len(ks[ks == actuals])
            result = common == len(ks) or common == 0

            print ks
            print actuals
            print s
            self.assertTrue(result)

    def test_fit_generated_data(self):
        # # k = 2 case
        # np.random.seed(2)
        # pis = np.array([.4,.6])
        # gammas = np.array([[10,1],
        #                    [1,10]])

        # print pis
        # print gammas
        # num_nodes = 40
        # data, ks = utils.generate_data(num_nodes, pis, gammas)
        # k = len(pis)

        # model = sbm.SBM(e_iterations=25)
        # best_pis = None
        # best_gammas = None
        # best_log_prob = -sys.maxint

        # for idx in range(5):

        #     cur_log_prob = model.fit(data, k, max_iterations=50, threshold=1e-3)
        #     if cur_log_prob > best_log_prob:
        #         best_log_prob = cur_log_prob
        #         best_pis = copy.deepcopy(model.pis)
        #         best_gammas = copy.deepcopy(model.gammas)

        # print best_log_prob
        # print best_pis
        # print best_gammas

        # k = 3 case
        np.random.seed(20)
        pis = np.array([.3,.2,.5])
        gammas = np.array([[10,15,20],
                           [15,25,30],
                           [20,30,35]])
        print pis
        print gammas
        num_nodes = 35
        data, ks = utils.generate_data(num_nodes, pis, gammas)
        k = len(pis)

        model = sbm.SBM(e_iterations=20)
        best_pis = None
        best_gammas = None
        best_log_prob = -sys.maxint

        for idx in range(5):

            cur_log_prob = model.fit(data, k, max_iterations=50, threshold=1e-3)
            if cur_log_prob > best_log_prob:
                best_log_prob = cur_log_prob
                best_pis = copy.deepcopy(model.pis)
                best_gammas = copy.deepcopy(model.gammas)

        print best_log_prob
        print best_pis
        print best_gammas

    def test_fit_real_data(self):
        np.random.seed(2)
        input_filepath = '../data/fungi_net.csv'
        data, keys = utils.load_data(input_filepath)
        model = sbm.SBM(e_iterations=2)

        num_k = 10
        num_runs = 1
        max_iterations = 10
        threshold = 1e-5

        output_filepath = '../data/results_fungi.npz'

        d = np.load('../data/results_fungi.npz')
        log_probs = list(d['log_probs'])
        icls = list(d['icls'])
        bics = list(d['bics'])
        pis = list(d['pis'])
        gammas = list(d['gammas'])

        # bics, icls, log_probs = [], [], []
        # pis, gammas = [], []
        for k in range(10, 11):
            best_pis, best_gammas = None, None
            best_log_prob, best_bic, best_icl = -sys.maxint, -sys.maxint, -sys.maxint
            for idx in range(num_runs):
                try:
                    log_prob, bic, icl = model.fit(data, k, max_iterations, threshold)
                except Exception as e:
                    print e
                    continue
                if icl > best_icl:
                    best_icl = icl
                    best_bic = bic
                    best_log_prob = log_prob
                    best_pis = copy.deepcopy(model.pis)
                    best_gammas = copy.deepcopy(model.gammas)
            pis.append(best_pis)
            gammas.append(best_gammas)
            bics.append(best_bic)
            icls.append(best_icl)
            log_probs.append(best_log_prob)
            # save after each k
            raw_input()
            np.savez(output_filepath, pis=pis, gammas=gammas, bics=bics, icls=icls, log_probs=log_probs)
        
            plt.plot(range(len(log_probs)), log_probs, label='log_probs')
            plt.title('log_probs')
            plt.savefig('../data/log_probs.png')
            plt.close()

            plt.plot(range(len(bics)), bics, label='bics')
            plt.title('bics')
            plt.savefig('../data/bics.png')
            plt.close()

            plt.plot(range(len(icls)), icls, label='icls')
            plt.title('icls')
            plt.savefig('../data/icls.png')
            plt.close()

        best_idx = np.argmax(icls)
        print 'best k: {}'.format(best_idx + 1)
        print 'pis: {}'.format(pis[best_idx])
        print 'gammas: {}'.format(gammas[best_idx])

def analyze_save():
    d = np.load('../data/results_fungi.npz')
    log_probs = d['log_probs']
    icls = d['icls']
    bics = d['bics']
    pis = d['pis']
    gammas = d['gammas']

    plt.plot(range(1, len(log_probs) + 1), log_probs, label='log_probs', linestyle='--')
    plt.plot(range(1, len(bics) + 1), bics, label='bics', linestyle='-')
    plt.plot(range(1, len(icls) + 1), icls, label='icls', linestyle='-.')
    plt.title('model selection')
    plt.legend(loc='best')
    plt.savefig('../data/model_selection.png')
    plt.close()

    best_idx = np.argmax(icls)
    print 'best k: {}'.format(best_idx + 1)
    print 'pis: {}'.format(pis[best_idx])
    print 'gammas: {}'.format(gammas[best_idx])

if __name__ == '__main__':
    analyze_save()
    # unittest.main()