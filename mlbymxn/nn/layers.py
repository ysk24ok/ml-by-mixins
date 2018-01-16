from abc import abstractmethod

import numpy as np

from ..activation_functions import (
    IdentityActivationMixin,
    SigmoidActivationMixin,
    TanhActivationMixin,
    ReLUActivationMixin
)
from ..loss_functions import LogLossMixin


class InputLayer(object):

    layer_type = 'input'

    def __init__(self, X):
        self.A = X
        self.size = X.shape[1]


class BaseLayer(object):

    eps = np.finfo(float).eps

    def __init__(self, layer_size: int, l2_reg: float=0):
        self.size = layer_size
        self.l2_reg = l2_reg
        self.theta = None
        self.A = None
        self.Z = None

    def forwardprop(self, theta, A_prev):
        """
        params
        ------
        theta: 2d array, shape (n_current+1, n_next)
               weight parameters of this layer
        A_prev: 2d array, shape (m, n_current+1)
                activated value of previous layer

        return
        ------
        A: 2d array, shape (m, n_next)
           activated output value of this layer
        Z: 2d array, shape (m, n_next)
           raw (non-activated) output value of this layer
        """
        Z = A_prev @ theta
        return self.activation(Z), Z


class BaseHiddenLayer(BaseLayer):

    layer_type = 'hidden'

    @abstractmethod
    def backprop(self, theta, A_prev, dLdA_next):
        pass

    def _theta_for_l2reg(self, theta):
        if len(theta.shape) == 1:
            return np.append(0, theta[1:])
        elif len(theta.shape) == 2:
            n = theta.shape[1]
            return np.append(np.zeros((1, n)), theta[1:], axis=0)


class BaseFullyConnectedLayer(BaseHiddenLayer):

    def backprop(self, theta, A_prev, dLdA_next, Z_cached=None):
        """
        params
        ------
        theta: 2d array, shape (n_current+1, n_next)
               weight parameters of this layer
        A_prev: 2d array, shape (m, n_current+1)
                activated output value of previous layer
        dLdA_next: 2d array, shape (m, n_next)
        Z_cached: 2d array, shape(m, n_next), None in default
                  Z calculated in forwardprop step

        return
        ------
        gradient: 2d array, shape (n_current+1, n_next)
        dLdA: 2d array, shape (m, n_current)
        """
        m = A_prev.shape[0]
        # Use cached Z if exists
        Z = Z_cached
        if Z is None:
            Z = A_prev @ theta
        dLdZ = dLdA_next * self.activation_gradient(Z)
        grad = A_prev.T @ dLdZ
        grad += self.l2_reg * self._theta_for_l2reg(theta)
        # exclude coefficients for bias term
        dLdA = (dLdZ @ theta.T)[:, 1:]
        return grad / m, dLdA


class FullyConnectedLayerIdentity(
        BaseFullyConnectedLayer, IdentityActivationMixin):

    pass


class FullyConnectedLayerSigmoid(
        BaseFullyConnectedLayer, SigmoidActivationMixin):

    pass


class FullyConnectedLayerTanh(
        BaseFullyConnectedLayer, TanhActivationMixin):

    pass


class FullyConnectedLayerReLU(
        BaseFullyConnectedLayer, ReLUActivationMixin):

    pass


class BaseOutputLayer(BaseLayer):

    layer_type = 'output'

    def backprop(self, theta, A_prev, Y, Z_cached=None, A_cached=None):
        """
        params
        ------
        theta: 2d array, shape (n_current+1, n_output)
               weight parameters of this layer
        A_prev: 2d array, shape (m, n_current+1)
                activated output value of previous layer
        Y: 2d array, shape (m, n_output)
        Z_cached: 2d array, shape(m, n_output), default is None
                  Z calculated in forwardprop step
        A_cached: 2d array, shape(m, n_output), default is None
                  A calculated in forwardprop step

        return
        ------
        gradient: 2d array, shape (n_current+1, n_output)
        dLdA: 2d array, shape (m, n_current)
        """
        m = A_prev.shape[0]
        # Use cached Z if exists
        Z = Z_cached
        if Z is None:
            Z = A_prev @ theta
        # Use cached A if exists
        A = A_cached
        if A is None:
            A = self.activation(Z)
        dLdZ = self._dLdZ(theta, Z, Y, A)
        grad = A_prev.T @ dLdZ
        grad += self.l2_reg * self._theta_for_l2reg(theta)
        # exclude coefficients for bias term
        dLdA = (dLdZ @ theta.T)[:, 1:]
        return grad / m, dLdA


class OutputLayerLogLoss(
        BaseOutputLayer, LogLossMixin, SigmoidActivationMixin):

    pass
