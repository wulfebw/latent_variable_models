
import copy
import math
import numpy as np

import utils

def log_factorial(value):
    return np.sum(np.log(v) for v in range(1, int(value) + 1, 1))

def log_poisson_density(point, mean):
    if mean <= 0:
        raise ValueError('mean value must be > 0, got : {}'.format(mean))
    return point * np.log(mean) - mean - log_factorial(point)

class SBM(object):

    def __init__(self, e_iterations=1):
        self.e_iterations = e_iterations

    def initialize(self, data, k):
        # track data to fit
        self.data = data

        # N = number of nodes
        self.N = data.shape[0]

        # k = number of latent classes
        self.k = k

        # allocate tau containers: num_nodes rows, num_classes columns
        self.taus = np.random.rand(self.N * self.k).reshape(self.N, self.k)
        self.taus /= np.sum(self.taus, axis=1, keepdims=True)

        # initialize parameters randomly
        # pi is the probability of each latent class
        self.pis = np.ones(self.k) / self.k
        # gamma is the mean of the poisson associated with edges 
        # between class i and class j
        self.gammas = np.abs(np.random.randn(self.k * self.k).reshape(self.k, self.k)) + np.max(self.data)

    def m_step(self):
        # assert valid tau
        assert np.all([abs(v - 1) < 1e-5 for v in np.sum(self.taus, axis=1)])
        assert not np.any([v < 0 for v in self.taus.flatten()])

        # update latent class probabilities
        self.pis = np.sum(self.taus, axis=0) / self.N

        # update connection probabilities
        for k in range(self.k):
            for l in range(self.k):
                total = 0
                normalizer = 0
                for i in range(self.N):
                    for j in range(self.N):
                        total += self.taus[i,k] * self.taus[j,l] * self.data[i,j]
                        normalizer += self.taus[i,k] * self.taus[j,l]
                self.gammas[k,l] = total / normalizer

    def e_step(self):
        # assert valid tau
        assert np.all([abs(v - 1) < 1e-5 for v in np.sum(self.taus, axis=1)])
        assert not np.any([v < 0 for v in self.taus.flatten()])

        # assert valid pi
        assert abs(np.sum(self.pis) - 1) < 1e-5 and not np.any([v < 0 for v in self.pis])

        # assert valid gammas
        assert not np.any([v < 0 for v in self.gammas.flatten()])

        # use deepcopy
        # start total at a value
        for idx in range(self.e_iterations):
            tau_copy = copy.deepcopy(self.taus)
            for i in range(self.N):
                for k in range(self.k):
                    total = np.log(self.pis[k])
                    for j in range(self.N):
                        if i == j: continue
                        for l in range(self.k):
                            edge_prob = log_poisson_density(self.data[i,j], self.gammas[k,l])
                            total += tau_copy[j,l] * edge_prob

                    self.taus[i,k] = total

            # normalize
            for i in range(self.N):
                self.taus[i, :] -= utils.log_sum_exp(self.taus[i, :]) 

            # exponentiate
            self.taus = np.exp(self.taus)

            # find residual
            residual = np.max(np.abs(self.taus - tau_copy))

            if residual < 1e-10:
                break

    def log_prob(self):
        # assert valid tau
        assert np.all([abs(v - 1) < 1e-5 for v in np.sum(self.taus, axis=1)])
        assert not np.any([v < 0 for v in self.taus.flatten()])

        # assert valid pi
        assert abs(np.sum(self.pis) - 1) < 1e-5 and not np.any([v < 0 for v in self.pis])

        # assert valid gammas
        assert not np.any([v < 0 for v in self.gammas.flatten()])

        # prior
        prob = np.sum(self.taus * np.log(self.pis).reshape(1, -1))

        # estimate of completed likelihood
        for i in range(self.N):
            for j in range(i + 1, self.N):
                for k in range(self.k):
                    for l in range(self.k):
                        density = log_poisson_density(self.data[i,j], self.gammas[k,l])
                        prob += self.taus[i,k] * self.taus[j,l] * density

        # return joint log probability for use in ICL
        joint_prob = prob
        # entropy
        prob += -np.sum(self.taus * np.log(self.taus))

        # assert valid prob
        assert not np.isnan(prob)

        return prob, joint_prob

    def fit(self, data, k, max_iterations, threshold, verbose=True):
        # allocate containers
        self.initialize(data, k)

        # repeatedly loop through e and m steps until convergence
        prev_log_prob = log_prob = log_joint_prob = 0
        for idx in range(max_iterations):
            # e-step
            self.e_step()

            # m-step
            self.m_step()

            # compute log_prob
            log_prob, log_joint_prob = self.log_prob()

            if verbose: 
                print 'iter: {}\tlog_prob: {:.4f}'.format(idx, log_prob)

            # check for convergence
            if abs(log_prob - prev_log_prob) < threshold: 
                break
            prev_log_prob = log_prob

        num_params = k - 1 + k * (k - 1)
        num_samples = len(data)
        bic = log_prob - num_params / 2. * np.log(num_samples) 
        icl = log_joint_prob - num_params / 2. * np.log(num_samples)
        return log_prob, bic, icl
