import numpy as np


class PCA(object):

    def __init__(self, dim: int):
        self.dim = dim

    def fit(self, X):
        m = len(X)
        #cov_matrix = np.cov(X.T)
        cov_matrix = X.T @ X / m
        self.U, self.S, self.V = np.linalg.svd(cov_matrix)

    def transform(self, X):
        return X @ self.U[:, 0:self.dim]

    def inverse_transform(self, X):
        return X @ self.U[:, 0:self.dim].T
