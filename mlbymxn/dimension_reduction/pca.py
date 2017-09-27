import numpy as np


class PCA(object):

    def __init__(self, dim: int):
        self.dim = dim

    def fit(self, X: np.array):
        m = len(X)
        #cov_matrix = np.cov(X.T)
        cov_matrix = np.dot(X.T, X) / m
        self.U, self.S, self.V = np.linalg.svd(cov_matrix)

    def transform(self, X: np.array):
        return np.dot(X, self.U[:, 0:self.dim])

    def inverse_transform(self, X: np.array):
        return np.dot(X, self.U[:, 0:self.dim].T)
