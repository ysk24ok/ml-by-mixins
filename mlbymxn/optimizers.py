from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.optimize import minimize


class BaseOptimizerMixin(object, metaclass=ABCMeta):

    def fit(self, X, y):
        m, n = X.shape
        # when theta is not initialized
        if len(self.theta) == 0:
            self.initialize_theta(np.random.rand(n) - 0.5)
        if self.verbose is True:
            cost = self.loss_function(self.theta, X, y)
            print('initial cost: {0:.6f}'.format(cost))
        num_iters = 1
        while True:
            self.update_theta(X, y)
            if self.verbose is True:
                cost = self.loss_function(self.theta, X, y)
                print('iter: {0}, cost: {1:.6f}'.format(num_iters, cost))
            if num_iters >= self.max_iters:
                break
            num_iters += 1

    @abstractmethod
    def update_theta(self, X, y):
        pass


class ScipyOptimizerMixin(object):

    def fit(self, X, y):
        m, n = X.shape
        # when theta is not initialized
        if len(self.theta) == 0:
            self.initialize_theta(np.random.rand(n) - 0.5)
        res = minimize(
            self.loss_function,
            self.theta,
            args=(X, y),
            method='L-BFGS-B',
            jac=self.gradient,
        )
        if res.success is not True:
            raise ValueError('Failed. status: {}, message: {}'.format(
                res.status, res.message))
        self.theta = res.x


class GradientDescentMixin(BaseOptimizerMixin):

    def update_theta(self, X, y):
        grad = self.gradient(self.theta, X, y)
        self.theta -= self.eta * grad


class StochasticGradientDescentMixin(BaseOptimizerMixin):

    def update_theta(self, X, y):
        m, n = X.shape
        indices = np.arange(m)
        if self.shuffle is True:
            np.random.shuffle(indices)
        num_sub_iters = 1
        while True:
            p_idx = self.batch_size * (num_sub_iters - 1)
            n_idx = self.batch_size * num_sub_iters
            idx = indices[p_idx:n_idx]
            X_partial = X[idx]
            y_partial = y[idx]
            if X_partial.shape[0] < self.batch_size:
                break
            grad = self.gradient(self.theta, X_partial, y_partial)
            self.theta -= self.eta * grad
            num_sub_iters += 1


class NewtonMixin(BaseOptimizerMixin):

    def update_theta(self, X, y):
        grad = self.gradient(self.theta, X, y)
        hessian = self.hessian(self.theta, X)
        self.theta -= self.eta * np.linalg.inv(hessian) @ grad
