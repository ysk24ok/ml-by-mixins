from abc import abstractmethod

import numpy as np


class BaseActivationFunctionMixin(object):

    @abstractmethod
    def activation_function(self, a):
        pass

    @abstractmethod
    def activation_function_gradient(self, z):
        """
        Note that z is an activated value
        """
        pass


class IdentityActivationFunctionMixin(BaseActivationFunctionMixin):

    def activation_function(self, a):
        return a

    def activation_function_gradient(self, z):
        return 1


class ExponentialActivationFunctionMixin(BaseActivationFunctionMixin):

    def activation_function(self, a):
        return np.exp(a)

    def activation_function_gradient(self, z):
        return z


class SigmoidActivationFunctionMixin(BaseActivationFunctionMixin):

    def activation_function(self, a):
        # Avoid 'RuntimeWarning: overflow encountered in exp'
        f = lambda x: 1 / (1 + np.exp(-x)) if -x < 500 else self.eps
        # Avoid 'RuntimeWarning: divide by zero encountered in log'
        return np.clip(np.vectorize(f)(a), a_min=self.eps, a_max=1-self.eps)

    def activation_function_gradient(self, z):
        return (1-z) * z


class StepActivationFunctionMixin(BaseActivationFunctionMixin):

    def activation_function(self, a, neg_label: int=0):
        return np.vectorize(lambda x: 1 if x >= 0 else neg_label)(a)

    def activation_function_gradient(self, z):
        # TODO
        pass
