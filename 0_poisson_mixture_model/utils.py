
import collections
import csv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

COLORS = ['red','blue','green','teal','yellow','magenta','orange','black','brown','pink']

def plot_data_k(data, k, means):
    """
    Plot original generated data, labeling using ordering convention.
    """
    means = np.asarray(means)
    samples_each = int(len(data) / k)
    for kidx, color in zip(range(k), COLORS):
        start = kidx * samples_each
        end = (kidx + 1) * samples_each
        plt.scatter(data[start:end, 0], data[start:end, 1], c=color)
    plt.scatter(means[:,0], means[:,1], c='green', marker='*', s=400)
    plt.show()

def plot_data_assigments(data, means, assignments):
    """
    Plot original generated data, labeling using assignments.
    """
    means = np.asarray(means)
    plt.scatter(data[:,0], data[:,1], c=assignments)
    plt.scatter(means[:,0], means[:,1], c='green', marker='*', s=400)
    plt.show()

def plot_1d_data_assigments(data, means, assignments):
    """
    Plot original generated data, labeling using assignments.
    """
    counter = collections.defaultdict(lambda: [0,-1])
    for sample, assignment in zip(data, assignments):
        k = sample[0]
        counter[k][0] += 1
        counter[k][1] = assignment

    for k, v in counter.iteritems():
        plt.bar(k, v[0], color=COLORS[v[1]], width=.5, alpha=.3)
    
    plt.scatter(means, np.zeros(len(means)), c='green', marker='*', s=400)
    plt.xlabel('counts')
    plt.ylabel('frequency')
    plt.show()

def plot_data_responsibilities(data, responsibilities, means):
    """
    Plt data colored based on class responsibilities.
    """
    k = np.shape(responsibilities)[-1]
    if k not in [2, 3]:
        msg = 'plot_data_responsibilities only works for 2 or 3 classes, got: {}'.format(k)
        raise ValueError(msg)

    # add a dimension of zeros if k = 2
    if k == 2:
        zs = np.zeros((len(responsibilities),1))
        responsibilities = np.hstack((responsibilities, zs))

    plt.scatter(data[:,0], data[:,1], facecolors=responsibilities)
    plt.scatter(means[:,0], means[:,1], c='green', marker='*', s=400)
    plt.show()

def plot_1d_data_responsibilities(data, responsibilities, means):
    """
    Plot original generated data, labeling using assignments.
    """
    num_classes = len(means)
    counter = collections.defaultdict(lambda: [0, np.zeros((num_classes))])
    for sample, sample_responsibilities in zip(data, responsibilities):
        k = sample[0]
        for responsibility in sample_responsibilities:
            counter[k][0] += 1
            counter[k][1][int(responsibility) - 1] += 1

    def get_color(weights):
        return 'red'

    for k, v in counter.iteritems():
        c = get_color(v[1])
        plt.bar(k, v[0], color=c, width=.5, alpha=.3)
    
    plt.scatter(means, np.zeros(len(means)), c='green', marker='*', s=400)
    plt.show()

def plot_sweep(log_probs, bics):
    """
    Plot model selection metrics resulting from a sweep.
    """
    k_range = range(len(log_probs))
    plt.plot(k_range, log_probs, label='log prob', c='red')
    plt.plot(k_range, bics, label='bic', c='blue')
    plt.legend()
    plt.show()

def load_1d_data(filepath):
    data = []
    with open(filepath, 'rb') as infile:
        infile.readline()
        csv_reader = csv.reader(infile, delimiter=',')
        for row in csv_reader:
            data.append(int(row[-1]))
    return np.asarray(data).reshape(-1,1)

def generate_data(k, num_samples, x_limits, y_limits):
    """
    Generate two-dimensional gaussian mixture data.
    """
    # allocate storage assuming equal dist of points
    samples_each = int(num_samples / k)
    num_samples = samples_each * k
    mixuture_data = np.empty((num_samples, 2))

    # generate data
    means = []
    for kidx in range(k):
        # choose means
        x = np.random.uniform(low=x_limits[0], high=x_limits[1])
        y = np.random.uniform(low=y_limits[0], high=y_limits[1])
        means.append([x, y])
        
        # generate random normal data
        data = np.random.normal(size=(samples_each, 2))
        data[:, 0] += x
        data[:, 1] += y

        # add to collection
        mixuture_data[kidx * samples_each: (kidx + 1) * samples_each, :] = data

    # return only means since using identiy as covariance
    return mixuture_data, means

def euclidean_dist(p1, p2):
    return np.sum((p1 - p2) ** 2)

def compute_total_dist(data, means, assignments):
    total_dist = 0
    for sample, assignment in zip(data, assignments):
        cur_mean = means[assignment]
        total_dist += euclidean_dist(sample, cur_mean)
    return total_dist

def normal(point, mean):
    diff = point - mean
    prob = 1. / np.sqrt(2. * np.pi) * np.exp(-np.dot(diff, diff) / 2.)
    return prob

def poisson(point, mean):
    return mean ** point * np.exp(-mean) / np.math.factorial(point)

def bic(data, log_prob, num_params):
    """
    Bayesian information criterion.
    """
    return -2 * log_prob + num_params * len(data)



