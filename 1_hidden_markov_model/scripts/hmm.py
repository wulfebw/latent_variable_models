
import numpy as np

def poisson_density(point, mean):
    return mean ** point * np.exp(-mean) / np.math.factorial(point)

class HMM(object):

    def __init__(self, data, k, max_iterations, threshold, verbose=True, seed=1):
        self.data = data
        self.k = k
        self.m, self.T = data.shape
        self.max_iterations = max_iterations
        self.threshold = threshold
        self.verbose = verbose
        np.random.seed(seed)

    def initialize(self):
        # unpack dimensions
        m, T, k = self.m, self.T, self.k

        # initialize model parameters
        unnormalized_pi = np.random.rand(k)
        self.pi = unnormalized_pi / np.sum(unnormalized_pi)
        unnormalized_A = np.random.rand(k, k) 
        self.A = unnormalized_A / np.sum(unnormalized_A, axis=1, keepdims=True)
        self.B = np.random.randint(low=0, high=np.max(self.data), size=(k))

        # allocate responsibilities containers
        self.alphas = np.empty((m, T + 1, k))
        self.betas = np.empty((m, T + 1, k))
        self.gammas = np.empty((m, T, k, k))

        # different try
        self.gammas = np.empty((m, T, k))
        self.etas = np.empty((m, T, k, k))

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
        self.alphas[:, 0, :] = self.pi

        # perform a forward pass for each sample
        for sidx, sample in enumerate(self.data):

            # tidx starts at 1 since zeroth timestep 
            # of alphas has already been initialized
            for tidx, value in enumerate(sample, 1):

                # iterate over k values to fill
                for j in range(self.k):

                    # iterate over previous k values
                    timestep_total = 0
                    for i in range(self.k):
                        emission_prob = poisson_density(value, self.B[j])
                        transition_prob = self.A[i, j]
                        alpha_prob = self.alphas[sidx, tidx - 1, i]
                        timestep_total += emission_prob * transition_prob * alpha_prob

                    # set value for jth class at time t
                    self.alphas[sidx, tidx, j] = timestep_total

    def backward(self):
        # initialize first timestep value of beta to one
        # and then iterate backward starting from the end
        self.betas[:, -1, :] = 1

        # perform a backward pass for each sample
        for sidx, sample in enumerate(self.data):

            # tidx starts at the last emission timestep (T - 1)
            # since timestep T has been filled with ones
            # and moves backward through time to timestep 0
            for tidx in range(self.T - 1, -1, -1):

                # iterate over k values to fill (timestep t)
                # note that i and j are flipped from forward pass
                for i in range(self.k):

                    # iterate over next k values (timestep t + 1)
                    timestep_total = 0
                    for j in range(self.k):
                        emission_prob = poisson_density(self.data[sidx, tidx], self.B[j])
                        transition_prob = self.A[i, j]
                        beta_prob = self.betas[sidx, tidx + 1, j]
                        timestep_total += emission_prob * transition_prob * beta_prob

                    # set value for jth class at time t
                    self.betas[sidx, tidx, i] = timestep_total

    def e_step(self):
        # compute alphas and betas
        self.forward()
        self.backward()

        # print self.A
        # print self.B
        # self.alphas = np.array([[[.5,.5],[.1,.2]]])
        # self.betas = np.array([[[.2,.1],[.3,.1]]])
        # raw_input()

        # # compute gammas
        # for sidx, sample in enumerate(self.data):
        #     for tidx in range(self.T):
        #         for i in range(self.k):
        #             for j in range(self.k):
        #                 a = self.alphas[sidx, tidx, i]
        #                 b = self.betas[sidx, tidx + 1, j]
        #                 transition_prob = self.A[i, j]
        #                 emission_prob = poisson_density(self.data[sidx, tidx], self.B[j])
        #                 self.gammas[sidx, tidx, i, j] = a * transition_prob * emission_prob * b
        #                 print a
        #                 print b
        #                 print transition_prob
        #                 print emission_prob
        #                 print self.gammas
        #                 raw_input()

        # # compute log probability of data
        # log_prob = np.sum(np.log(self.gammas + 1e-8))
        # return log_prob

        # gammas
        # self.gammas = self.alphas[:,1:,:] * self.betas[:,:-1,:] 
        # self.gammas = np.mean(self.gammas, axis=0)
        # self.gammas = self.gammas / np.sum(self.gammas, axis=(1))

        # # etas
        # for sidx, sample in enumerate(self.data):
        #     for tidx in range(self.T):
        #         for i in range(self.k):
        #             for j in range(self.k):
        #                 a = self.alphas[sidx, tidx, i]
        #                 b = self.betas[sidx, tidx + 1, j]
        #                 transition_prob = self.A[i, j]
        #                 emission_prob = poisson_density(self.data[sidx, tidx], self.B[j])
        #                 self.etas[sidx, tidx, i, j] = a * transition_prob * emission_prob * b

        # self.etas = np.mean(self.etas, axis=0)
        # self.eta /= np.sum(self.alphas[:, -1, :], axis=(-1))
        # print
        # raw_input()

    def m_step(self):
        # average over samples
        alphas = np.mean(self.alphas, axis=0)
        betas = np.mean(self.betas, axis=0)

        print alphas
        print betas
        raw_input()

        # update A
        for i in range(self.k):
            for j in range(self.k):
                for t in range(self.T):
                    pass



        # # update A
        # # sum over samples and timesteps
        # new_A = np.sum(self.gammas, axis=(0,1))
        # # normalize across classes by dividing each column by its sum
        # self.A = new_A / (np.sum(new_A, axis=0, keepdims=True) + 1e-8)

        # # update pi
        # # ???

        # # update B
        # # recompute mean value
        # self.B = np.zeros(self.k)
        # for i in range(self.k):
        #     for j in range(self.k):
        #         for sidx, sample in enumerate(self.data):
        #             for tidx, value in enumerate(sample):
        #                 self.B[i] += self.gammas[sidx, tidx, i, j] * value

        # # normalize
        # self.B /= np.sum(self.gammas, axis=(0,1,3))

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

