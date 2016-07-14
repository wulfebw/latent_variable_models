
import numpy as np
import sys

import utils


def e_step(data, means, phis):
    """
    Args:
        - data: samples as row, shape (num_samples, dim)
            (for poisson dim = 1)
        - means: current mean values, shape (k)
        - phis: current class probabilities, shape (k)
    """
    # compute $p(x|z) * p(z)$ using current parameters
    w = np.empty((len(data), len(means)))
    for sidx, sample in enumerate(data):
        for midx, mean in enumerate(means): 
            prior = phis[midx]
            # exp(log_poisson(.)) for numerical stability
            x_given_z = np.exp(log_poisson(sample[0], mean))
            w[sidx, midx] = prior * x_given_z

    # compute $p(z|x) = p(x|z) * p(z) / p(x)$
    # sum of each row is $p(x)$
    # so by dividing by the sum of each row we get $p(z|x)$
    row_sum_w = np.sum(w, axis=1, keepdims=True)
    w /= row_sum_w

    # also compute log probability $log(p(x))$
    log_prob = np.sum(np.log(row_sum_w))
    return w, log_prob

def e_step(data, means, phis, density):
    w = np.empty((len(data), len(means)))
    for sidx, sample in enumerate(data):
        for midx, mean in enumerate(means): 
            prior = phis[midx]

            if density == utils.log_poisson:
                x_given_z = np.exp(density(sample[0], mean))
            else:
                x_given_z = density(sample, mean)

            w[sidx, midx] = prior * x_given_z

    row_sum_w = np.sum(w, axis=1, keepdims=True)
    w /= row_sum_w

    log_prob = np.sum(np.log(row_sum_w))
    return w, log_prob

def sweep(data, model, max_k=10, verbose=True):

    log_probs, bics, icls = [], [], []
    best_responsibilities, best_means, best_phis, best_k = None, None, None, 0
    best_bic = -sys.maxint
    for k in range(1, max_k + 1):
        means, phis, log_prob, responsibilities = model(data, k, verbose=False)

        # k classes, each with 1 additional associated param
        num_params = 2 * k

        # bic 
        bic = utils.bic(data, log_prob, num_params)

        # icl = bic minus entropy of responsibilities
        icl = bic + utils.entropy(responsibilities)

        log_probs.append(log_prob)
        bics.append(bic)
        icls.append(icl)

        if bic > best_bic:
            best_k = k
            best_bic = bic
            best_responsibilities = responsibilities
            best_means = means
            best_phis = phis

        if verbose:
            print '\nk: {}\tlog_prob: {:.5f}\tbic: {:.5f}'.format(k, log_prob, bic)
            print 'phis: {}'.format(phis)
            print 'means: {}'.format(means)
            # utils.plot_1d_data_responsibilities(data, responsibilities, means)
            # utils.plot_data_responsibilities(data, responsibilities, means)

    return best_k, log_probs, bics, icls, best_responsibilities, best_means, best_phis

    