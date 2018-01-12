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

    def activation(self, z):
        return z

    def activation_gradient(self, z):
        return 1


class ExponentialActivationMixin(BaseActivationMixin):

    def activation(self, z):
        return np.exp(z)

    def activation_gradient(self, z):
        return self.activation(z)


class SigmoidActivationMixin(BaseActivationMixin):

    def activation(self, z):
        # Avoid 'RuntimeWarning: overflow encountered in exp'
        f = lambda x: 1 / (1 + np.exp(-x)) if -x < 500 else self.eps
        # Avoid 'RuntimeWarning: divide by zero encountered in log'
        return np.clip(np.vectorize(f)(z), a_min=self.eps, a_max=1-self.eps)

    def activation_gradient(self, z):
        a = self.activation(z)
        return (1-a) * a


class StepActivationMixin(BaseActivationMixin):

    def activation(self, z, neg_label: int=0):
        return np.vectorize(lambda x: 1 if x >= 0 else neg_label)(z)

    def activation_gradient(self, z):
        # TODO
        pass
