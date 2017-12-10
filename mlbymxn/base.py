import numpy as np


class BaseML(object):

    eps = np.finfo(float).eps

    def __init__(
            self, max_iters: int=100, eta: float=0.01, l2_reg: float=0.0,
            verbose: bool=False, shuffle: bool=True, batch_size: int=1):
        # maximum number of iterations
        self.max_iters = max_iters
        # eta: learning rate
        self.eta = eta
        # theta: weight vector whose shape is (n,)
        self.theta = np.array([])
        # l2_reg: coefficient for L2 regularization term
        self.l2_reg = l2_reg
        # verbosity flag
        self.verbose = verbose
        # shuffle training samples every iteration
        # NOTE: only used for online-fashion optimizer
        self.shuffle = shuffle
        # number of training samples to be used for each (forward) pass
        # batch_size=1   -> SGD
        # 1<batch_size<m -> mini-batch SGD
        # batch_size=m   -> GD
        # NOTE: only used for online-fashion optimizer
        self.batch_size = batch_size

    def initialize_theta(self, initial_theta):
        # NOTE: bias term should be contained in initial_theta
        self.theta = initial_theta
