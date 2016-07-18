
import numpy as np

def generate_data(A, B, pi, T, emission_dist, num_samples):
    """
    K = number of latent classes
    N = number of observation classes when applicable

    args:
        - A: transition matrix, shape (K, K)
        - B: emission matrix
            + for poisson: shape (K) of poisson means
            + for multinomial: shape (K, N)
        - pi: initial state probabilities, shape (K)
        - T: number of timesteps in each sample
        - emission_dist: emission distribution 
        - num_samples: total number of samples to generate
    """
    # collect the observations, not the hidden states
    data = np.empty((num_samples, T))
    for sidx in range(num_samples):

        # initial state sample
        z = np.argmax(np.random.multinomial(1, pi))

        # transition for each timestep
        for tidx in range(T):

            # sample hidden state at tidx
            z = np.argmax(np.random.multinomial(1, A[z, :]))

            # sample observation at tidx
            y = emission_dist(B[z, :])

            # record sample
            data[sidx, tidx] = y

    return data
            
if __name__ =='__main__':

    A = np.array([[.9,.1],[.2,.8]])
    B = np.array([[10],[5]])
    pi = np.array([.5,.5])
    T = 2
    dist = np.random.poisson
    num_samples = 30
    data = generate_data(A, B, pi, T, dist, num_samples)
    print data