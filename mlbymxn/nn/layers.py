import numpy as np

from ..activation_functions import (
    IdentityActivationFunctionMixin,
    SigmoidActivationFunctionMixin
)
from ..loss_functions import LogLossMixin


class InputLayer(object):

    layer_type = 'input'

    def __init__(self, matrix):
        self.matrix = matrix
        self.size = matrix.shape[1]


class BaseLayer(object):

    eps = np.finfo(float).eps

    def __init__(self, layer_size: int, l2_reg: float=0):
        self.size = layer_size
        self.l2_reg = l2_reg
        self.matrix = None
        self.theta = None


class BaseHiddenLayer(BaseLayer):

    layer_type = 'hidden'

    def forwardprop(self, theta, X):
        pass

    def gradient(self, theta, X):
        pass

    def backprop(self, theta, X):
        pass

    def _theta_for_l2reg(self, theta):
        if len(theta.shape) == 1:
            return np.append(0, theta[1:])
        elif len(theta.shape) == 2:
            n = theta.shape[1]
            return np.append(np.zeros((1, n)), theta[1:], axis=0)


class BaseFullyConnectedLayer(BaseHiddenLayer):

    def forwardprop(self, theta, X):
        """
        params
        ------
        theta:      n_current+1 x n_next
        X:          m x n_current+1

        return
        ------
        output:     m x n_next
        """
        return self.activation_function(X @ theta)

    def gradient(self, theta, X, backprop):
        """
        params
        ------
        theta:      n_current+1 x n_next
        X:          m x n_current+1
        backprop:   m x n_next

        return
        ------
        gradient:   n_current+1 x n_next
        """
        m = X.shape[0]
        z = self.forwardprop(theta, X)
        dLdA = backprop * self.activation_function_gradient(z)
        grad = X.T @ dLdA
        grad += self.l2_reg * self._theta_for_l2reg(theta)
        return grad / m

    def backprop(self, theta, X, backprop):
        """
        params
        ------
        theta:      n_current+1 x n_next
        X:          m x n_current+1
        backprop:   m x n_next

        return
        ------
        backprop:   m x n_current
        """
        z = self.forwardprop(theta, X)
        dLdA = backprop * self.activation_function_gradient(z)
        # exclude coefficients for bias term
        return (dLdA @ theta.T)[:, 1:]


class FullyConnectedLayerIdentity(
        BaseFullyConnectedLayer, IdentityActivationFunctionMixin):

    pass


class FullyConnectedLayerSigmoid(
        BaseFullyConnectedLayer, SigmoidActivationFunctionMixin):

    pass


class BaseOutputLayer(BaseLayer):

    layer_type = 'output'

    def forwardprop(self, theta, X):
        return self.predict(theta, X)


class OutputLayerLogLoss(
        BaseOutputLayer, LogLossMixin, SigmoidActivationFunctionMixin):

    pass
