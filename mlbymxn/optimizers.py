from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.optimize import minimize


class BaseOptimizerMixin(object, metaclass=ABCMeta):

    def before_fit(self, X, y):
        pass

    def after_fit(self, X, y):
        pass

    def fit(self, X, y):
        # initialize theta when it is not initialized yet
        if len(self.theta) == 0:
            self.initialize_theta(X)
        if self.verbose is True:
            cost = self.loss_function(self.theta, X, y)
            print('initial cost: {0:.6f}'.format(cost))
        self.before_fit(X, y)
        num_iters = 1
        while True:
            self.update_theta(X, y)
            if self.verbose is True:
                cost = self.loss_function(self.theta, X, y)
                print('iter: {0}, cost: {1:.6f}'.format(num_iters, cost))
            if num_iters >= self.max_iters:
                break
            num_iters += 1
        self.after_fit(X, y)

    @abstractmethod
    def update_theta(self, X, y):
        pass


class ScipyOptimizerMixin(object):

    def fit(self, X, y):
        # initialize theta when it is not initialized yet
        if len(self.theta) == 0:
            self.initialize_theta(X)
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


class GDOptimizerMixin(BaseOptimizerMixin):

    def update_theta(self, X, y):
        grad = self.gradient(self.theta, X, y)
        self.theta -= self.eta * grad


class SGDOptimizerMixin(BaseOptimizerMixin):

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


class MomentumSGDOptimizerMixin(BaseOptimizerMixin):

    def before_fit(self, X, y):
        self.prev_update_term = np.zeros((len(self.theta),))

    def update_theta(self, X, y):
        m, n = X.shape
        indices = np.arange(m)
        if self.shuffle is True:
            np.random.shuffle(indices)
        num_sub_iters = 1
        for idx in indices:
            p_idx = self.batch_size * (num_sub_iters - 1)
            n_idx = self.batch_size * num_sub_iters
            idx = indices[p_idx:n_idx]
            X_partial = X[idx]
            y_partial = y[idx]
            if X_partial.shape[0] < self.batch_size:
                break
            grad = self.gradient(self.theta, X_partial, y_partial)
            #update_term = self.momentum * self.prev_update_term + (1 - self.momentum) * grad
            update_term = self.momentum * self.prev_update_term + grad
            self.theta -= self.eta * update_term
            self.prev_update_term = update_term
            num_sub_iters += 1


class SAGOptimizerMixin(BaseOptimizerMixin):

    def before_fit(self, X, y):
        m, n = X.shape
        self.sum_grad = np.zeros((n,))
        self.sum_counts = 0
        self.latest_errors = np.zeros((m,))

    def update_theta(self, X, y):
        m, n = X.shape
        indices = np.arange(m)
        if self.shuffle is True:
            np.random.shuffle(indices)
        num_sub_iters = 1
        for idx in indices:
            X_partial = X[idx].reshape((1, n))
            y_partial = y[idx].reshape((1,))
            grad = self.gradient(self.theta, X_partial, y_partial)
            current_err = grad[1] / X[idx][1]
            prev_err = self.latest_errors[idx]
            self.sum_grad += X[idx] * (current_err - prev_err)
            self.latest_errors[idx] = current_err
            self.sum_counts += 1
            # In first iteration, SAG doesn't update theta,
            # just calculates prev_err of all samples
            if self.sum_counts > m:
                self.theta -= self.eta * self.sum_grad / m
            #else:
                #self.theta -= self.eta * self.sum_grad / self.sum_counts
            num_sub_iters += 1


class NewtonOptimizerMixin(BaseOptimizerMixin):

    def update_theta(self, X, y):
        grad = self.gradient(self.theta, X, y)
        hessian = self.hessian(self.theta, X)
        self.theta -= self.eta * np.linalg.inv(hessian) @ grad
