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
        return loss[0][0]

    def gradient(self, X: np.array, y: np.array) -> np.array:
        m = len(X)
        err = self.predict(X) - y       # m x 1
        return np.dot(X.T, err) / m     # n x 1

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
        return loss[0][0]

    def gradient(self, X: np.array, y: np.array) -> np.array:
        m = len(X)
        err = self.predict(X) - y       # m x 1
        return np.dot(X.T, err) / m     # n x 1

    def hessian(self, X: np.array) -> np.array:
        m = len(X)
        prob = self._sigmoid(np.dot(X, self.theta))         # m x 1
        return np.dot(prob.T, prob) * np.dot(X.T, X) / m    # n x n
