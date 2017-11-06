from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.misc import factorial


class BaseLossMixin(object, metaclass=ABCMeta):

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def loss_function(self, X, y):
        pass

    @abstractmethod
    def gradient(self, X, y):
        pass


class SquaredLossMixin(BaseLossMixin):

    def predict(self, theta, X):
        return X @ theta

    def loss_function(self, theta, X, y) -> float:
        m = X.shape[0]
        err = self.predict(theta, X) - y
        loss = np.sum(err ** 2)
        loss += self.l2_reg * np.sum(theta[1:] ** 2)
        return loss / (2 * m)

    def gradient(self, theta, X, y):
        m, n = X.shape
        err = self.predict(theta, X) - y
        grad = X.T @ err
        grad += self.l2_reg * np.append(0, theta[1:])
        return grad / m

    def hessian(self, theta, X):
        m = X.shape[0]
        return X.T @ X / m


class LogLossMixin(BaseLossMixin):

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict(self, theta, X):
        return self._sigmoid(X @ theta)

    def loss_function(self, theta, X, y) -> float:
        m = X.shape[0]
        prob = self.predict(theta, X)
        # TODO: prevent 'RuntimeWarning: divide by zero encountered in log'
        loss_pos = y @ np.log(prob)
        loss_neg = (1-y) @ np.log(1-prob)
        loss = - (loss_pos + loss_neg) / m
        loss += self.l2_reg * np.sum(theta[1:] ** 2) / (2 * m)
        return loss

    def gradient(self, theta, X, y):
        m = X.shape[0]
        err = self.predict(theta, X) - y
        grad = X.T @ err
        grad += self.l2_reg * np.append(0, theta[1:])
        return grad / m

    def hessian(self, theta, X):
        m = X.shape[0]
        prob = self._sigmoid(X @ theta)
        return prob @ prob * X.T @ X / m


class HingeLossMixin(BaseLossMixin):

    def predict(self, theta, X):
        z = X @ theta
        return np.vectorize(lambda x: 1 if x >= 0 else -1)(z)

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
        return np.exp(X @ theta)

    def loss_function(self, theta, X, y) -> float:
        m = X.shape[0]
        z = self.predict(theta, X)
        loss = -np.sum(-z + X @ theta * y - np.log(factorial(y))) / m
        loss += self.l2_reg * np.sum(theta[1:] ** 2) / (2 * m)
        return loss

    def gradient(self, theta, X, y):
        m = X.shape[0]
        err = self.predict(theta, X) - y
        grad = X.T @ err
        grad += self.l2_reg * np.append(0, theta[1:])
        return grad / m

    def hessian(self, theta, X):
        pass
