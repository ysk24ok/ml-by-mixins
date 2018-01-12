from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.misc import factorial


class BaseLossMixin(object, metaclass=ABCMeta):

    @abstractmethod
    def predict(self, theta, X):
        pass

    @abstractmethod
    def loss_function(self, theta, X, y):
        pass

    @abstractmethod
    def gradient(self, theta, X, y):
        pass

    def _theta_for_l2reg(self, theta):
        if len(theta.shape) == 1:
            return np.append(0, theta[1:])
        elif len(theta.shape) == 2:
            n = theta.shape[1]
            return np.append(np.zeros((1, n)), theta[1:], axis=0)


class SquaredLossMixin(BaseLossMixin):

    def predict(self, theta, X):
        return self.activation(X @ theta)

    def loss_function(self, theta, X, y) -> float:
        m = X.shape[0]
        err = self.predict(theta, X) - y
        loss = np.sum(err ** 2)
        loss += self.l2_reg * np.sum(self._theta_for_l2reg(theta) ** 2)
        return loss / (2 * m)

    def gradient(self, theta, X, y):
        m = X.shape[0]
        z = X @ theta
        err = (self.activation(z) - y) * self.activation_gradient(z)
        #err = self.predict(theta, X) - y
        grad = X.T @ err
        grad += self.l2_reg * self._theta_for_l2reg(theta)
        return grad / m

    def backprop(self, theta, X, y):
        z = X @ theta
        err = (self.activation(z) - y) * self.activation_gradient(z)
        #err = self.predict(theta, X) - y
        return (err @ theta.T)[:, 1:]

    def hessian(self, theta, X):
        m = X.shape[0]
        return X.T @ X / m


class LogLossMixin(BaseLossMixin):

    def predict(self, theta, X):
        return self.activation(X @ theta)

    def loss_function(self, theta, X, y) -> float:
        m = X.shape[0]
        a = self.predict(theta, X)
        loss_pos = y * np.log(a)
        loss_neg = (1-y) * np.log(1-a)
        loss = - np.sum(loss_pos + loss_neg) / m
        loss += self.l2_reg * np.sum(self._theta_for_l2reg(theta) ** 2) / (2 * m)
        return loss

    def gradient(self, theta, X, y):
        m = X.shape[0]
        z = X @ theta
        a = self.activation(z)
        err = (- y / a + (1-y) / (1-a)) * self.activation_gradient(z)
        #err = self.predict(theta, X) - y
        grad = X.T @ err
        grad += self.l2_reg * self._theta_for_l2reg(theta)
        return grad / m

    def backprop(self, theta, X, y):
        z = X @ theta
        a = self.activation(z)
        err = (- y / a + (1-y) / (1-a)) * self.activation_gradient(z)
        #err = self.predict(theta, X) - y
        return (err @ theta.T)[:, 1:]

    def hessian(self, theta, X):
        m = X.shape[0]
        z = self.activation(X @ theta)
        return z @ z * X.T @ X / m


# TODO: l2_reg
class HingeLossMixin(BaseLossMixin):

    def predict(self, theta, X):
        # TODO: neg_label=-1 only if it's Perceptron
        #       maybe neg_label should be a property of ML class
        return self.activation(X @ theta, neg_label=-1)

    def _loss(self, theta, X, y):
        m = X.shape[0]
        z = X @ theta * y
        # max(0, t-ywx)
        return np.maximum(np.zeros((m,)), self.threshold - z)

    def loss_function(self, theta, X, y) -> float:
        return np.sum(self._loss(theta, X, y))

    def gradient(self, theta, X, y):
        loss = self._loss(theta, X, y)
        # max(0, -yx)
        grad = -(X.T * loss) @ y
        return grad


class PoissonLossMixin(BaseLossMixin):

    def predict(self, theta, X):
        return self.activation(X @ theta)

    def loss_function(self, theta, X, y) -> float:
        m = X.shape[0]
        z = self.predict(theta, X)
        loss = -np.sum(-z + X @ theta * y - np.log(factorial(y))) / m
        loss += self.l2_reg * np.sum(self._theta_for_l2reg(theta) ** 2) / (2 * m)
        return loss

    def gradient(self, theta, X, y):
        m = X.shape[0]
        z = X @ theta
        err = - (y / self.activation(z) - 1) * self.activation_gradient(z)
        #err = self.predict(theta, X) - y
        grad = X.T @ err
        grad += self.l2_reg * self._theta_for_l2reg(theta)
        return grad / m

    def backprop(self, theta, X, y):
        z = self.predict(theta, X)
        err = - (y / z - 1) * self.activation_gradient(z)
        #err = z - y
        return (err @ theta.T)[:, 1:]

    def hessian(self, theta, X):
        pass
