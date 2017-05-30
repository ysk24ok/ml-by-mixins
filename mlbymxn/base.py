import numpy as np


class BaseML(object):

    def __init__(
            self, max_iters: int=100, eta: float=0.01, l2_reg: float=0.0,
            verbose: bool=False):
        # maximum numebr of iterations
        self.max_iters = max_iters
        # eta: learning rate
        self.eta = eta
        # theta: weight vector, n x 1
        self.theta = np.array([])
        # l2_reg: coefficient for L2 regularization
        self.l2_reg = l2_reg
        # verbosity flag
        self.verbose = verbose

    def initialize_theta(self, initial_theta: np.array([])):
        # NOTE: bias term should be contained in initial_theta
        self.theta = initial_theta
