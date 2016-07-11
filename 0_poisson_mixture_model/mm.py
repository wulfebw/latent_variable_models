
import numpy as np
import sys

import utils

def e_step(data, means, phis, density):
    w = np.empty((len(data), len(means)))
    for sidx, sample in enumerate(data):
        for midx, mean in enumerate(means):
            prior = phis[midx]
            x_given_z = density(sample, mean)
            w[sidx, midx] = prior * x_given_z

    row_sum_w = np.sum(w, axis=1, keepdims=True)
    w /= row_sum_w
    log_prob = np.sum(np.log(row_sum_w))
    return w, log_prob

def sweep(data, model, max_k=10, verbose=True):

    log_probs, bics = [], []
    best_responsibilities, best_means, best_phis, best_k = None, None, None, 0
    best_bic = sys.maxint
    for k in range(1, max_k + 1):
        means, phis, log_prob, responsibilities = model(data, k, verbose=True)
        bic = utils.bic(data, log_prob, num_params=2 * k)

        log_probs.append(log_prob)
        bics.append(bic)

        if bic < best_bic:
            best_k = k
            best_bic = bic
            best_responsibilities = responsibilities
            best_means = means
            best_phis = phis

        if verbose:
            print '\nk: {}\tlog_prob: {:.5f}\tbic: {:.5f}'.format(k, log_prob, bic)
            print 'phis: {}'.format(phis)
            print 'means: {}'.format(means)

    return best_k, log_probs, bics, best_responsibilities, best_means, best_phis

    