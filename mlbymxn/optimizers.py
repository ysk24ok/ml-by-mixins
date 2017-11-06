from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.optimize import minimize
from sklearn.utils import shuffle


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
            self.update_theta(self.theta, X, y)
            if self.verbose is True:
                cost = self.loss_function(self.theta, X, y)
                print('iter: {0}, cost: {1:.6f}'.format(num_iters, cost))
            if num_iters >= self.max_iters:
                break
            num_iters += 1

    @abstractmethod
    def update_theta(self, theta, X, y):
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

    def update_theta(self, theta, X, y):
        grad = self.gradient(theta, X, y)
        self.theta -= self.eta * grad


class StochasticGradientDescentMixin(BaseOptimizerMixin):

    def update_theta(self, theta, X, y):
        m, n = X.shape
        # shuffle X and y while maintaining the correspondance of each sample
        if self.shuffle is True:
            X_shuffled, y_shuffled = shuffle(X, y)
        num_sub_iters = 1
        while True:
            p = self.batch_size * (num_sub_iters - 1)
            n = self.batch_size * num_sub_iters
            X_partial = X[p:n]
            y_partial = y[p:n]
            if len(X_partial) < self.batch_size:
                break
            grad = self.gradient(theta, X_partial, y_partial)
            self.theta -= self.eta * grad
            num_sub_iters += 1


class NewtonMixin(BaseOptimizerMixin):

    def update_theta(self, theta, X, y):
        grad = self.gradient(theta, X, y)
        hessian = self.hessian(theta, X)
        self.theta -= self.eta * np.linalg.inv(hessian) @ grad
