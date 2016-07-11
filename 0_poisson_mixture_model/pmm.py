
import matplotlib.pyplot as plt
import numpy as np
import sys

import initialization
import mm
import utils

def m_step(data, k, w):
    dim = data.shape[-1]

    # compute phis
    col_sum_w = np.sum(w, axis=0)
    phis = col_sum_w / len(data)
    
    # compute means
    means = np.zeros((k, dim))
    for sidx, sample in enumerate(data):
        for midx in range(k):
            means[midx] += w[sidx, midx] * sample

    for midx in range(k):
        means[midx, :] /= col_sum_w[midx]

    return means, phis

def pmm(data, k, max_iterations=20, threshold=1e-4, verbose=False):
    """
    Fits a poisson mixture model to the data with k latent classes.

    Args:
        - data: data to assign to different mixtures, shape (num_samples, dim)
        - k: number of latent classes

    Return Values:
        - means: values of means params, shape (k, dim)
        - phis: probs of different classes, shape (k)
        - log_prob: last computed log probability of data
        - w: responsibilities of different classes for each sample, shape (num_samples, k)
    """
    # initialize the mean parameters for each class
    means, assignments, _ = initialization.initialize(data, k, num_runs=15)
    
    if verbose:
        utils.plot_1d_data_assigments(data, means, assignments)

    # assume balanced dataset
    phis = np.ones(k) / float(k)

    # repeatedly run e-step and m-step until convergence
    # initialize some values to reuse or return
    prev_log_prob, log_prob, w = 0, -1, None
    for idx in range(max_iterations):

        # e-step
        w, log_prob = mm.e_step(data, means, phis, density=utils.poisson)

        # m-step
        means, phis = m_step(data, k, w)

        # check for convergence
        diff = abs(log_prob - prev_log_prob)
        if diff < threshold:
            break
        else:
            prev_log_prob = log_prob

            if verbose:
                # check how it's going
                utils.plot_1d_data_responsibilities(data, w, means)
                print 'log prob: {:.5f}\tchange in log prob: {:.5f}'.format(log_prob, diff)
            
    return means, phis, log_prob, w

if __name__ == '__main__':
    # load data
    filepath = 'data/poisson_mixture.csv'
    data = utils.load_1d_data(filepath)

    # sweep over k values to find best model
    best_k, log_probs, bics, best_responsibilities, best_means, best_phis = mm.sweep(data, pmm, max_k=5, verbose=True)

    best_log_prob = log_probs[best_k - 1]
    best_bic = bics[best_k - 1]
    print '\nbest k: {}\tlog_prob: {:.5f}\tbic: {:.5f}'.format(best_k, best_log_prob, best_bic)
    print 'phis: {}'.format(best_phis)
    print 'means: {}'.format(best_means)
    utils.plot_1d_data_responsibilities(data, best_responsibilities, best_means)
    utils.plot_sweep(log_probs, bics)

