from abc import ABCMeta, abstractmethod

import numpy as np


class BaseLossMixin(object, metaclass=ABCMeta):

    @abstractmethod
    def loss_function(self):
        pass

    @abstractmethod
    def gradient(self):
        pass


class SquaredLossMixin(BaseLossMixin):

    def predict(self, X: np.array) -> np.array:
        return np.dot(X, self.theta)

    def loss_function(self, X: np.array, y: np.array) -> float:
        m = len(X)
        err = self.predict(X) - y   # m x 1
        loss = np.dot(err.T, err) / (2 * m)
        loss += self.l2_reg * np.sum(self.theta[1:] ** 2) / (2 * m)
        return loss[0][0]

    def gradient(self, X: np.array, y: np.array) -> np.array:
        m = len(X)
        err = self.predict(X) - y       # m x 1
        grad = np.dot(X.T, err) / m     # n x 1
        grad += self.l2_reg * np.append(np.zeros((1, 1)), self.theta[1:], axis=0) / m
        return grad

    def hessian(self, X: np.array) -> np.array:
        m = len(X)
        return np.dot(X.T, X) / m   # n x n


class LogLossMixin(BaseLossMixin):

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict(self, X: np.array) -> np.array:
        return self._sigmoid(np.dot(X, self.theta))

    def loss_function(self, X: np.array, y: np.array) -> float:
        m = len(X)
        prob = self.predict(X)      # m x 1
        loss_pos = np.dot(y.T, np.log(prob))
        loss_neg = np.dot((1-y).T, np.log(1-prob))
        loss = - (loss_pos + loss_neg) / m
        loss += self.l2_reg * np.sum(self.theta[1:] ** 2) / (2 * m)
        return loss[0][0]

    def gradient(self, X: np.array, y: np.array) -> np.array:
        m = len(X)
        err = self.predict(X) - y       # m x 1
        grad = np.dot(X.T, err) / m     # n x 1
        grad += self.l2_reg * np.append(np.zeros((1, 1)), self.theta[1:], axis=0) / m
        return grad

    def hessian(self, X: np.array) -> np.array:
        m = len(X)
        prob = self._sigmoid(np.dot(X, self.theta))         # m x 1
        return np.dot(prob.T, prob) * np.dot(X.T, X) / m    # n x n


class HingeLossMixin(BaseLossMixin):

    def predict(self, X: np.array) -> np.array:
        z = np.dot(X, self.theta)   # m x 1
        return np.vectorize(lambda x: 1 if x >= 0 else -1)(z)

    def _loss(self, X: np.array, y: np.array) -> np.array:
        m = len(X)
        z = np.dot(X, self.theta) * y   # m x 1
        # max(0, 1-ywx)
        return np.maximum(np.zeros((m, 1)), self.threshold - z)

    def loss_function(self, X: np.array, y: np.array) -> float:
        return np.sum(self._loss(X, y))

    def gradient(self, X: np.array, y: np.array) -> np.array:
        m, n = X.shape
        loss = self._loss(X, y)     # m x 1
        # max(0, -yx)
        grad = -y * X * (loss > 0)      # m x n
        return np.sum(grad, axis=0).reshape((n, 1)) / m
