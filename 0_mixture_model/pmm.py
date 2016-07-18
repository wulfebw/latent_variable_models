
import matplotlib.pyplot as plt
import numpy as np
import sys

import initialization
import mm
import utils

def m_step(data, k, w):
    """
    Args:
        - data: samples as row, shape (num_samples, dim)
        - k: number of clusters / classes
        - w: responsibilities computed in e-step, shape (num_samples, K)
    """
    # phis
    # for each phi, sum its responsibilities over the dataset 
    # and divide by the number of samples total
    # formally: $phi_{l} = 1/m * sum_{i} w_{l}^{i}$
    col_sum_w = np.sum(w, axis=0)
    phis = col_sum_w / len(data)
    
    # means
    # for each class, compute a weighted sum of the samples 
    # and normalize by the total weight for the class
    # $mean_{l} = sum_{i} w_{l}^{i} * x^{i} / sum_{i} w_{l}^{i}$
    means = np.zeros(k)
    for sidx, sample in enumerate(data):
        for midx in range(k):
            means[midx] += w[sidx, midx] * sample
    means /= col_sum_w

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
    phis = np.zeros(k)
    for assignment in assignments:
        phis[assignment] += 1
    phis /= len(assignments)

    if verbose:
        utils.plot_1d_data_assigments(data, means, assignments)

    # repeatedly run e-step and m-step until convergence
    # initialize some values to reuse or return
    prev_log_prob, log_prob, w = 0, -1, None
    for idx in range(max_iterations):

        # e-step
        w, log_prob = mm.e_step(data, means, phis, density=utils.log_poisson)

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
    data = utils.load_1d_data(filepath, preprocess=False)

    data = data[:50]

    # sweep over k values to find best model
    best_k, log_probs, bics, icls, best_responsibilities, best_means, best_phis = mm.sweep(data, pmm, max_k=10, verbose=True)

    best_log_prob = log_probs[best_k - 1]
    best_bic = bics[best_k - 1]
    best_icl = icls[best_k - 1]
    print '\nbest k: {}\tlog_prob: {:.5f}\tbic: {:.5f}\ticl: {:.5f}'.format(best_k, best_log_prob, best_bic, best_icl)
    print 'log probs: {}'.format(log_probs)
    print 'bics: {}'.format(bics)
    print 'phis: {}'.format(best_phis)
    print 'means: {}'.format(best_means)

    utils.plot_1d_data_responsibilities(data, best_responsibilities, best_means)
    utils.plot_sweep(log_probs, bics, icls)

