
import numpy as np

def generate_data(A, B, pi, T, emission_dist):
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
    """
    # collect the observations, not the hidden states
    data = np.empty(T)

    # initial state sample
    z = np.argmax(np.random.multinomial(1, pi))
    y = emission_dist(B[z, :])
    data[0] = y

    # transition for each timestep
    for tidx in range(1, T):

        # sample hidden state at tidx
        z = np.argmax(np.random.multinomial(1, A[z, :]))

        # sample observation at tidx
        y = emission_dist(B[z, :])

        # record sample
        data[tidx] = y

    return data
            
if __name__ =='__main__':

    A = np.array([[0,1],[1,0]])
    B = np.array([[1],[10]])
    pi = np.array([.5,.5])
    T = 20
    dist = np.random.poisson
    data = generate_data(A, B, pi, T, dist)
    print data