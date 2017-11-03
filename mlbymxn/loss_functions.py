from abc import ABCMeta, abstractmethod

import numpy as np


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

    def predict(self, X):
        return X @ self.theta

    def loss_function(self, X, y) -> float:
        m = len(X)
        err = self.predict(X) - y   # m x 1
        loss = err.T @ err
        loss += self.l2_reg * np.sum(self.theta[1:] ** 2)
        return loss[0][0] / (2 * m)

    def gradient(self, X, y):
        m = len(X)
        err = self.predict(X) - y       # m x 1
        grad = X.T @ err        # n x 1
        grad += self.l2_reg * np.append(np.zeros((1, 1)), self.theta[1:], axis=0)
        return grad / m

    def hessian(self, X):
        m = len(X)
        return X.T @ X / m   # n x n


class LogLossMixin(BaseLossMixin):

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict(self, X):
        return self._sigmoid(X @ self.theta)

    def loss_function(self, X, y) -> float:
        m = len(X)
        prob = self.predict(X)      # m x 1
        loss_pos = y.T @ np.log(prob)
        loss_neg = (1-y).T @ np.log(1-prob)
        loss = - (loss_pos + loss_neg) / m
        loss += self.l2_reg * np.sum(self.theta[1:] ** 2) / (2 * m)
        return loss[0][0]

    def gradient(self, X, y):
        m = len(X)
        err = self.predict(X) - y       # m x 1
        grad = X.T @ err            # n x 1
        grad += self.l2_reg * np.append(np.zeros((1, 1)), self.theta[1:], axis=0)
        return grad / m

    def hessian(self, X):
        m = len(X)
        prob = self._sigmoid(X @ self.theta)        # m x 1
        return prob.T @ prob * X.T @ X / m    # n x n


class HingeLossMixin(BaseLossMixin):

    def predict(self, X):
        z = X @ self.theta      # m x 1
        return np.vectorize(lambda x: 1 if x >= 0 else -1)(z)

    def _loss(self, X, y):
        m = len(X)
        z = X @ self.theta * y   # m x 1
        # max(0, 1-ywx)
        return np.maximum(np.zeros((m, 1)), self.threshold - z)

    def loss_function(self, X, y) -> float:
        return np.sum(self._loss(X, y))

    def gradient(self, X, y):
        m, n = X.shape
        loss = self._loss(X, y)     # m x 1
        # max(0, -yx)
        grad = -y * X * (loss > 0)      # m x n
        return np.sum(grad, axis=0).reshape((n, 1)) / m
