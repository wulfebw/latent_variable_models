
import numpy as np

def poisson_density(point, mean):
    return mean ** point * np.exp(-mean) / np.math.factorial(point)

def log_factorial(value):
    return np.sum(np.log(v) for v in range(1, int(value) + 1, 1))

def log_poisson_density(point, mean):
    return point * np.log(mean) - mean - log_factorial(point)

class HMM(object):

    def __init__(self, data, k, max_iterations, threshold, verbose=True, seed=1):
        self.data = data
        self.k = k
        self.T = data.shape[0]
        self.max_iterations = max_iterations
        self.threshold = threshold
        self.verbose = verbose
        np.random.seed(seed)

    def initialize(self):
        # unpack dimensions
        T, k = self.T, self.k

        # initialize model parameters
        unnormalized_pi = np.random.rand(k)
        self.pi = unnormalized_pi / np.sum(unnormalized_pi)
        unnormalized_A = np.random.rand(k, k) 
        self.A = unnormalized_A / np.sum(unnormalized_A, axis=1, keepdims=True)
        self.B = np.random.randint(low=0, high=np.max(self.data), size=(k))

        # allocate responsibilities containers
        self.alphas = np.empty((T, k))
        self.betas = np.empty((T, k))

        # wiki
        self.gammas = np.empty((T, k))
        self.etas = np.empty((T, k, k))

    def forward(self):
        """
        The forward pass computes for each sample, for each timestep, 
        and for each latent class, the probability that the latent state
        was the latent class, and stores these values in self.alphas.
        
        This is accomplished using a dynamic programming approach that takes
        advantage of the assumption that the future depends only upon the previous 
        timestep. Specifically, it iterates through each sequence keeping track
        of the probability of each class up until that timestep. Then, to compute
        the probability of each time step at t + 1, it sums over a set of 
        probabilities where each is the probability of transitioning from a 
        previous class times the probability of the current class given the 
        observation times the probability of the previous class. This sum gives
        the total probability of being in a certain class at timestep t + 1.
        """
        # initialize first timestep value of alpha for each sample 
        # to the start probability of the corresponding latent class in A
        self.alphas[0, :] = np.log(self.pi)
        for i in range(self.k):
            self.alphas[0, i] += log_poisson_density(self.data[0], self.B[i])

        # tidx starts at 1 since zeroth timestep 
        # of alphas has already been initialized
        for tidx, value in enumerate(self.data[1:], 1):

            # iterate over k values to fill
            for j in range(self.k):

                # iterate over previous k values
                timestep_total = 0
                for i in range(self.k):
                    
                    transition_prob = self.A[i, j]
                    alpha_prob = np.exp(self.alphas[tidx - 1, i])
                    timestep_total += transition_prob * alpha_prob

                emission_prob = log_poisson_density(value, self.B[j])

                # set value for jth class at time t
                self.alphas[tidx, j] = np.log(timestep_total) + emission_prob

        # convert to normal from log form
        self.alphas = np.exp(self.alphas)

    def backward(self):
        # initialize first timestep value of beta to one
        # and then iterate backward starting from the end
        self.betas[-1, :] = 1

        # start from second to last
        for tidx in range(self.T - 2, -1, -1):

            # iterate over k values to fill (timestep t)
            # note that i and j are flipped from forward pass
            for i in range(self.k):

                # iterate over next k values (timestep t + 1)
                timestep_total = 0
                for j in range(self.k):
                    emission_prob = poisson_density(self.data[tidx + 1], self.B[j])
                    transition_prob = self.A[i, j]
                    beta_prob = self.betas[tidx + 1, j]
                    timestep_total += emission_prob * transition_prob * beta_prob

                # set value for jth class at time t
                self.betas[tidx, i] = timestep_total

    def e_step(self):
        # compute alphas and betas
        self.forward()
        self.backward()

        # gammas
        self.gammas = self.alphas * self.betas
        self.gammas = self.gammas / np.sum(self.gammas, axis=1, keepdims=True)

        # etas
        for tidx in range(self.T - 1):
            for i in range(self.k):
                for j in range(self.k):
                    a = self.alphas[tidx, i]
                    b = self.betas[tidx + 1, j]
                    transition_prob = self.A[i, j]
                    emission_prob = poisson_density(self.data[tidx + 1], self.B[j])
                    self.etas[tidx, i, j] = a * transition_prob * emission_prob * b

        self.etas /= np.sum(self.alphas[-1, :])

        return np.sum(np.log(self.alphas[-1]))

    def m_step(self):
        # pi
        self.pi = self.gammas[0, :]

        # transition probabilities
        for i in range(self.k):
            for j in range(self.k):
                numerator = 0
                denom = 0
                for tidx in range(self.T - 1):
                    numerator += self.etas[tidx, i, j]
                    denom += self.gammas[tidx, i]
                self.A[i, j] = numerator / denom

        # emission probabilities
        for i in range(self.k):
            total = 0
            denom = 0
            for tidx, value in enumerate(self.data):
                total += value * self.gammas[tidx, i]
                denom += self.gammas[tidx, i]

            self.B[i] = total / denom

    def fit(self):
        
        # initialize parameter estimates
        self.initialize()

        # run e_step, m_step for max iterations or until convergence
        prev_log_prob = log_prob = 0
        for idx in range(self.max_iterations):

            # e-step
            log_prob = self.e_step()

            # check for convergence
            diff = abs(log_prob - prev_log_prob)
            if diff < self.threshold:
                break
            else:
                prev_log_prob = log_prob
                if self.verbose:
                    print 'iter: {}\tlog_prob: {:.4f}\tdiff: {:.4f}'.format(idx, log_prob, diff)

            # m-step
            self.m_step()

        # return the log probability of the fit
        return log_prob

