
import csv
import matplotlib.pyplot as plt
import numpy as np

def plot_data(edges, classes):
    pass

def generate_data(num_nodes, pis, gammas):

    # generate latent class for each node
    ks = np.empty(num_nodes, dtype=int)
    for nidx in range(num_nodes):
        ks[nidx] = int(np.argmax(np.random.multinomial(1, pis, 1)))

    # generate edges for each node
    edges = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(i, num_nodes):
            edges[i, j] = np.random.poisson(gammas[ks[i],ks[j]])
    edges = (edges + edges.T).astype(int)

    return edges, ks

def log_sum_exp(values):
    max_value = max(values)
    if np.isinf(max_value):
        return -np.inf

    total = 0
    for v in values:
        total += np.exp(v - max_value)

    return np.log(total) + max_value

def load_data(input_filepath):
    data = []
    keys = []
    with open(input_filepath, 'rb') as infile:
        # remove header
        infile.readline()
        csv_reader = csv.reader(infile)
        for row in csv_reader:
            keys.append(row[0])
            values = [int(v) for v in row[1:]]
            data.append(values)

    data = np.asarray(data, dtype=np.float64)
    return data, keys

if __name__ == '__main__':
    input_filepath = '../data/tree_net.csv'
    data, keys = load_data(input_filepath)
    print np.diag(data)