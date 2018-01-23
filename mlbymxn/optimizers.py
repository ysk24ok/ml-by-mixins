from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.optimize import minimize


class BaseOptimizerMixin(object, metaclass=ABCMeta):

    def before_fit(self, X, y):
        pass

    def after_fit(self, X, y):
        pass

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def update_theta(self, X, y):
        pass


class BatchOptimizerMixin(BaseOptimizerMixin):

    def fit(self, X, y):
        # initialize theta when it is not initialized yet
        if len(self.theta) == 0:
            self.initialize_theta(X)
        if self.verbose is True:
            cost = self.loss_function(self.theta, X, y)
            print('iter: 0, cost: {0:.6f}'.format(cost))
        if self.warm_start is False:
            self.before_fit(X, y)
        num_iters = 1
        while True:
            self.update_theta(X, y)
            if self.verbose is True:
                cost = self.loss_function(self.theta, X, y)
                print('iter: {0}, cost: {1:.6f}'.format(num_iters, cost))
            if num_iters >= self.max_iters:
                break
            self.warm_start = True
            num_iters += 1
        self.after_fit(X, y)


class MinibatchOptimizerMixin(BaseOptimizerMixin):

    def fit(self, X, y):
        m = X.shape[0]
        # initialize theta when it is not initialized yet
        if len(self.theta) == 0:
            self.initialize_theta(X)
        if self.verbose is True:
            cost = self.loss_function(self.theta, X, y)
            print('iter: 0, cost: {0:.6f}'.format(cost))
        if self.warm_start is False:
            self.before_fit(X, y)
        num_iters = 1
        while True:     # epoch
            indices = np.arange(m)
            if self.shuffle is True:
                np.random.shuffle(indices)
            num_sub_iters = 1
            while True:     # iteration
                p_idx = self.batch_size * (num_sub_iters - 1)
                n_idx = self.batch_size * num_sub_iters
                idx = indices[p_idx:n_idx]
                X_minibatch = X[idx]
                y_minibatch = y[idx]
                if X_minibatch.shape[0] < self.batch_size:
                    break
                self.update_theta(X_minibatch, y_minibatch)
                num_sub_iters += 1
            if self.verbose is True:
                cost = self.loss_function(self.theta, X, y)
                print('iter: {0}, cost: {1:.6f}'.format(num_iters, cost))
            if num_iters >= self.max_iters:
                break
            self.warm_start = True
            num_iters += 1
        self.after_fit(X, y)


class ScipyOptimizerMixin(BaseOptimizerMixin):

    optimizer_type = 'scipy'

    def fit(self, X, y):
        options = {'maxiter': self.max_iters}
        # initialize theta when it is not initialized yet
        if len(self.theta) == 0:
            self.initialize_theta(X)
        if self.warm_start is False:
            self.before_fit(X, y)
        if self.verbose is True:
            options['disp'] = True
        res = minimize(
            self.loss_function,
            self.theta,
            args=(X, y),
            method='L-BFGS-B',
            jac=self.gradient,
            options=options
        )
        if res.success is not True:
            raise ValueError('Failed. status: {}, message: {}'.format(
                res.status, res.message))
        self.warm_start = True
        self.theta = res.x
        self.after_fit(X, y)

    def update_theta(self, X, y):
        pass


class GDOptimizerMixin(BatchOptimizerMixin):

    optimizer_type = 'gd'

    def update_theta(self, X, y):
        grad = self.gradient(self.theta, X, y)
        self.theta -= self.eta * grad


class NewtonOptimizerMixin(BatchOptimizerMixin):

    optimizer_type = 'newton'

    def update_theta(self, X, y):
        grad = self.gradient(self.theta, X, y)
        hessian = self.hessian(self.theta, X)
        self.theta -= self.eta * np.linalg.inv(hessian) @ grad


class SAGOptimizerMixin(BatchOptimizerMixin):

    optimizer_type = 'sag'

    def before_fit(self, X, y):
        m = X.shape[0]
        n = self.theta.shape[0]
        self.sum_grad = np.zeros((n,))
        self.sum_counts = 0
        # TODO: need too much memory
        self.latest_grad = np.zeros((m, n))

    def update_theta(self, X, y):
        m = X.shape[0]
        indices = np.arange(m)
        if self.shuffle is True:
            np.random.shuffle(indices)
        num_sub_iters = 1
        while True:
            p_idx = self.batch_size * (num_sub_iters - 1)
            n_idx = self.batch_size * num_sub_iters
            idx = indices[p_idx:n_idx]
            X_minibatch = X[idx]
            y_minibatch = y[idx]
            if X_minibatch.shape[0] < self.batch_size:
                break
            grad = self.gradient(self.theta, X_minibatch, y_minibatch)
            self.sum_grad -= self.latest_grad[idx[0]]
            self.sum_grad += grad
            self.latest_grad[idx[0]] = grad
            self.sum_counts += 1
            # In first iteration, SAG doesn't update theta,
            # just calculates prev_err of all samples
            if self.sum_counts > m:
                self.theta -= self.eta * self.sum_grad / m
            #else:
                #self.theta -= self.eta * self.sum_grad / self.sum_counts
            num_sub_iters += 1


class SGDOptimizerMixin(MinibatchOptimizerMixin):

    optimizer_type = 'sgd'

    def update_theta(self, X, y):
        grad = self.gradient(self.theta, X, y)
        self.theta -= self.eta * grad


class MomentumSGDOptimizerMixin(MinibatchOptimizerMixin):

    optimizer_type = 'momentum_sgd'

    def before_fit(self, X, y):
        self.v = np.zeros(self.theta.shape)

    def update_theta(self, X, y):
        grad = self.gradient(self.theta, X, y)
        self.v *= self.momentum
        self.v += (1 - self.momentum) * grad
        self.theta -= self.eta * self.v


class RMSpropOptimizerMixin(MinibatchOptimizerMixin):

    optimizer_type = 'rmsprop'

    def before_fit(self, X, y):
        self.s = np.zeros(self.theta.shape)

    def update_theta(self, X, y):
        grad = self.gradient(self.theta, X, y)
        self.s *= self.rmsprop_alpha
        self.s += (1 - self.rmsprop_alpha) * np.square(grad)
        grad *= (np.sqrt(self.s) + self.epsilon) ** -1
        self.theta -= self.eta * grad


class AdaGradOptimizerMixin(MinibatchOptimizerMixin):

    optimizer_type = 'adagrad'

    def before_fit(self, X, y):
        self.v = np.zeros(self.theta.shape)

    def update_theta(self, X, y):
        grad = self.gradient(self.theta, X, y)
        self.v += np.square(grad)
        grad *= (np.sqrt(self.v) + self.epsilon) ** -1
        self.theta -= self.eta * grad


class AdaDeltaOptimizerMixin(MinibatchOptimizerMixin):

    optimizer_type = 'adadelta'

    def before_fit(self, X, y):
        self.v = np.zeros(self.theta.shape)
        self.s = np.zeros(self.theta.shape)

    def update_theta(self, X, y):
        grad = self.gradient(self.theta, X, y)
        self.v *= self.adadelta_rho
        self.v += (1 - self.adadelta_rho) * np.square(grad)
        grad *= np.sqrt(self.s + self.epsilon)
        grad *= np.sqrt(self.v + self.epsilon) ** -1
        self.s *= self.adadelta_rho
        self.s += (1 - self.adadelta_rho) * np.square(grad)
        self.theta -= self.eta * grad


class AdamOptimizerMixin(MinibatchOptimizerMixin):

    optimizer_type = 'adam'

    def before_fit(self, X, y):
        self.v = np.zeros(self.theta.shape)
        self.s = np.zeros(self.theta.shape)
        self.t = 1

    def update_theta(self, X, y):
        grad = self.gradient(self.theta, X, y)
        self.v *= self.adam_beta1
        self.v += (1 - self.adam_beta1) * grad
        self.s *= self.adam_beta2
        self.s += (1 - self.adam_beta2) * np.square(grad)
        v_corrected = self.v * ((1 - np.power(self.adam_beta1, self.t)) ** -1)
        s_corrected = self.s * ((1 - np.power(self.adam_beta2, self.t)) ** -1)
        self.t += 1
        v_corrected *= (np.sqrt(s_corrected) + self.epsilon) ** -1
        self.theta -= self.eta * v_corrected
