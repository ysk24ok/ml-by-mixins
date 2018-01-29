from abc import abstractmethod

import numpy as np


class BaseActivationMixin(object):

    @abstractmethod
    def activation(self, z):
        pass

    @abstractmethod
    def activation_gradient(self, z):
        pass


class IdentityActivationMixin(BaseActivationMixin):

    activation_type = 'identity'

    def activation(self, z):
        return z

    def activation_gradient(self, z):
        return 1


class ExponentialActivationMixin(BaseActivationMixin):

    activation_type = 'exponential'

    def activation(self, z):
        return np.exp(z)

    def activation_gradient(self, z):
        return self.activation(z)


class SigmoidActivationMixin(BaseActivationMixin):

    activation_type = 'sigmoid'

    def activation(self, Z):
        # calculate 1 / (1 + np.exp(-Z)) if implemented naively
        # To avoid 'RuntimeWarning: overflow encountered in exp'
        Z_clipped = np.clip(Z, -700, np.finfo(float).max)
        A = 1 + np.exp(-Z_clipped)
        A **= -1
        eps = np.finfo(float).eps
        # To avoid 'RuntimeWarning: divide by zero encountered in log'
        return np.clip(A, eps, 1-eps)

    def activation_gradient(self, Z):
        A = self.activation(Z)
        return (1-A) * A


class TanhActivationMixin(BaseActivationMixin):

    activation_type = 'tanh'

    def activation(self, z):
        return np.tanh(z)

    def activation_gradient(self, z):
        return 1 - np.square(np.tanh(z))


class ReLUActivationMixin(BaseActivationMixin):

    activation_type = 'relu'

    def activation(self, Z):
        return np.maximum(Z, 0)

    def activation_gradient(self, Z):
        Z_copied = np.copy(Z)
        Z_copied[Z_copied >= 0.0] = 1.0
        return np.clip(Z_copied, 0, np.finfo(Z_copied.dtype).max)
