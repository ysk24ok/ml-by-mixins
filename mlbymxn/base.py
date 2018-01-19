import numpy as np


class BaseML(object):

    def __init__(
            self, max_iters: int=100, eta: float=0.01,
            initialization_type: str='normal', l2_reg: float=0.0,
            verbose: bool=False, shuffle: bool=True, batch_size: int=1,
            momentum: float=0.9, use_naive_impl: bool=False):
        # maximum number of iterations
        self.max_iters = max_iters
        # eta: learning rate
        self.eta = eta
        # weight initialization type
        self.initialization_type = initialization_type
        # weight vector
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
        # momentum term
        # NOTE: only used in MomentumSGDOptimizer
        self.momentum = momentum
        # True only in debug or unittest
        self.use_naive_impl = use_naive_impl

    def initialize_theta(self, X):
        self._initialize_theta(X.shape[1])

    def _initialize_theta(self, n: int):
        # NOTE: bias term should be contained in n
        if self.initialization_type == 'normal':
            self.theta = np.random.randn(n)
        elif self.initialization_type == 'uniform':
            self.theta = np.random.rand(n) - 0.5
        elif self.initialization_type == 'zero':
            self.theta = np.zeros((n,))
        elif self.initialization_type == 'one':
            self.theta = np.ones((n,))
        else:
            raise ValueError('Undefined weight initialization type: {}'.format(
                self.initialization_type))
