import numpy as np


class BaseML(object):

    def __init__(
            self, max_iters: int=100, eta: float=None,
            initialization_type: str='normal', l2_reg: float=0.0,
            verbose: bool=False, shuffle: bool=True, batch_size: int=1,
            momentum: float=0.9, rmsprop_alpha: float=0.99,
            adadelta_rho: float=0.95,
            adam_beta1: float=0.9, adam_beta2: float=0.999, epsilon: float=1e-8,
            use_naive_impl: bool=False):
        # maximum number of iterations
        self.max_iters = max_iters
        # eta: learning rate
        self._set_eta(eta)
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
        # exponential weight decay rate used in MomentumSGDOptimizer
        self.momentum = momentum
        # exponential weight decay rate used in RMSpropOptimizer
        self.rmsprop_alpha = rmsprop_alpha
        # exponential weight decay rate used in AdaDeltaOptimizer
        self.adadelta_rho = adadelta_rho
        # exponential weight decay rate used in AdamOptimizer
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        # small value for numerical stability
        self.epsilon = epsilon
        # True only in debug or unittest
        self.use_naive_impl = use_naive_impl

    def _set_eta(self, eta: float) -> float:
        default_eta = 0.01
        if eta is not None:
            self.eta = eta
            return
        if hasattr(self, 'optimizer_type') is False:
            self.eta = default_eta
            return
        if self.optimizer_type == 'adagrad':
            self.eta = 0.001
        elif self.optimizer_type == 'adadelta':
            self.eta = 1
        elif self.optimizer_type == 'adam':
            self.eta = 0.001
        else:
            self.eta = default_eta

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
