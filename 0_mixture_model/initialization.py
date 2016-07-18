
import collections
import numpy as np
import sys

import utils

def k_means(data, k, max_iterations=100):
    """
    Runs k-means to get initial values
    """
    if len(data) < k:
        raise ValueError('must have more data points than centers, got: {}'.format(len(data)))

    dim = 1 if len(np.shape(data)) == 1 else np.shape(data)[-1]
    means = np.empty((k, dim))

    # initialize means to random points in dataset
    taken = set()
    for kidx in range(k):
        rand_idx = np.random.randint(len(data))
        while rand_idx in taken:
            rand_idx = np.random.randint(len(data))
        taken.add(rand_idx)
        means[kidx, :] = data[rand_idx, :]

    prev_assignments = []
    assignments = []
    for idx in range(max_iterations):

        # check for convergence
        if prev_assignments != [] and prev_assignments == assignments:
            break

        # reset assignments
        prev_assignments = assignments
        assignments = []

        # assign to means
        for sample in data:
            min_k_idx = 0
            min_k_dist = utils.euclidean_dist(means[0], sample)
            for kidx in range(1, k):
                k_dist = utils.euclidean_dist(means[kidx], sample)
                if k_dist < min_k_dist:
                    min_k_dist = k_dist
                    min_k_idx = kidx
            assignments.append(min_k_idx)

        # show intermediate cluster assignments
        # utils.plot_1d_data_assigments(data, means, assignments)

        # recompute means
        means = np.zeros((k, dim))
        mean_counts = collections.defaultdict(int)
        for sample, kidx in zip(data, assignments):
            means[kidx, :] += sample
            mean_counts[kidx] += 1
        
        for kidx, count in mean_counts.iteritems():
            means[kidx] /= float(count)
        
    return means, assignments

def initialize(data, k, num_runs):
    """
    Given the data to fit the mixture model to, initialize the parameters of the
    model using heuristics.

    Args:
        - k: number of classes
        - data: np.array of shape (num_samples, 2)

    Return Values:
        - means: np.array of shape (k, 2)
    """
    # run k-means a few times and take best
    means = []
    assignments = []
    best_dist = sys.maxint

    for _ in range(num_runs):
        cur_means, cur_assignments = k_means(data, k)
        dist = utils.compute_total_dist(data, cur_means, cur_assignments)

        if dist < best_dist:
            best_dist = dist
            means = cur_means
            assignments = cur_assignments

    return means, assignments, best_dist

def run_k_means_gmm():
    k = 3
    num_samples = 1000
    x_limits = [0,10]
    y_limits = [0,10]
    data, means = utils.generate_data(k=k, num_samples=num_samples, x_limits=x_limits, y_limits=y_limits)

    # plot generated data
    utils.plot_data_k(data, k, means)

    # run k-means a few times and take best
    init_means, assignments, best_dist = initialize(data, k, num_runs=10)

    # plot results
    print 'total euclidean distance: {}'.format(best_dist)
    utils.plot_data_assigments(data, init_means, assignments)

def run_k_means_poisson():
    # load data
    filepath = 'data/poisson_mixture.csv'
    data = utils.load_1d_data(filepath, preprocess=True)

    # run k_means
    k = 3
    init_means, assignments, best_dist = initialize(data, k, num_runs=10)
    
    # plot results
    print 'total euclidean distance: {}'.format(best_dist)
    utils.plot_1d_data_assigments(data, init_means, assignments)

if __name__ == '__main__':
    run_k_means_poisson()


