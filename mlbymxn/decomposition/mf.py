import numpy as np

from ..base import BaseML
from ..activation_functions import IdentityActivationMixin
from ..loss_functions import SquaredLossMixin
from ..optimizers import SGDOptimizerMixin


class MatrixFactorization(BaseML, SquaredLossMixin, IdentityActivationMixin):

    def __init__(self, hidden_dim: int, **kargs):
        self.hidden_dim = hidden_dim
        super(MatrixFactorization, self).__init__(**kargs)

    def initialize_theta(self, X):
        self.num_users, self.num_items = np.max(X, axis=0)
        n = self.hidden_dim * self.num_users + self.hidden_dim * self.num_items
        super(MatrixFactorization, self)._initialize_theta(n)

    def predict(self, theta, X):
        # TODO
        # P: user matrix, Q: item matrix
        p_dim = self.hidden_dim * self.num_users
        P = theta[:p_dim].reshape((self.num_users, self.hidden_dim))
        Q = theta[p_dim:].reshape((self.num_items, self.hidden_dim))
        return np.array([P[X[i][0]-1, :] @ Q[X[i][1]-1, :].T for i in range(X.shape[0])])

    def gradient(self, theta, X, y) -> float:
        # TODO: activation function
        # NOTE: based on the premise that optimizer is online-fashion
        assert X.shape[0] == 1 and y.shape[0] == 1
        err = self.predict(theta, X) - y
        user_id = X[0][0] - 1
        item_id = X[0][1] - 1
        gradient = np.zeros(theta.shape[0])
        user_s_idx = user_id * self.hidden_dim
        user_e_idx = (user_id + 1) * self.hidden_dim
        p_dim = self.hidden_dim * self.num_users
        item_s_idx = p_dim + item_id * self.hidden_dim
        item_e_idx = p_dim + (item_id + 1) * self.hidden_dim
        u_vec = theta[user_s_idx:user_e_idx]
        i_vec = theta[item_s_idx:item_e_idx]
        gradient[user_s_idx:user_e_idx] = err[0] * i_vec + self.l2_reg * u_vec
        gradient[item_s_idx:item_e_idx] = err[0] * u_vec + self.l2_reg * i_vec
        return gradient


class MatrixFactorizationSGD(MatrixFactorization, SGDOptimizerMixin):

    pass
