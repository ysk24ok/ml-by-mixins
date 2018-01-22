from typing import List

import numpy as np

from .layers import (
    InputLayer,
    FullyConnectedLayerSigmoid,
    FullyConnectedLayerTanh,
    FullyConnectedLayerReLU,
    OutputLayerLogLoss
)
from ..base import BaseML
from ..optimizers import (
    ScipyOptimizerMixin,
    GDOptimizerMixin,
    SAGOptimizerMixin,
    SGDOptimizerMixin,
    MomentumSGDOptimizerMixin,
    RMSpropOptimizerMixin,
    AdaGradOptimizerMixin,
    AdaDeltaOptimizerMixin,
    AdamOptimizerMixin,
)
from ..utils import add_bias


class MultiLayerPerceptron(BaseML):

    def __init__(
            self, hidden_layer_sizes: List[int], output_layer_size: int,
            activation: str='sigmoid', **kargs):
        # activation for hidden layers
        if activation == 'sigmoid':
            hidden_layer = FullyConnectedLayerSigmoid
        elif activation == 'tanh':
            hidden_layer = FullyConnectedLayerTanh
        elif activation == 'relu':
            hidden_layer = FullyConnectedLayerReLU
        else:
            raise ValueError('Undefined activation: {}'.format(activation))
        # generate layers
        l2_reg = kargs['l2_reg'] if 'l2_reg' in kargs else 0.0
        use_naive_impl = kargs['use_naive_impl'] if 'use_naive_impl' in kargs else False
        self.hidden_layers = []
        for size in hidden_layer_sizes:
            self.hidden_layers.append(
                hidden_layer(size, l2_reg=l2_reg, use_naive_impl=use_naive_impl))
        self.output_layer = OutputLayerLogLoss(
            output_layer_size, l2_reg=l2_reg, use_naive_impl=use_naive_impl)
        super(MultiLayerPerceptron, self).__init__(**kargs)

    def initialize_theta(self, X):
        prev_layer_size = X.shape[1]
        layers = self.hidden_layers + [self.output_layer]
        n = 0
        for current_layer in layers:
            current_theta_size = prev_layer_size * current_layer.size
            n += current_theta_size
            prev_layer_size = current_layer.size + 1    # bias term
        super(MultiLayerPerceptron, self)._initialize_theta(n)

    def _theta_for_l2reg(self, theta):
        if len(theta.shape) == 1:
            return np.append(0, theta[1:])
        elif len(theta.shape) == 2:
            n = theta.shape[1]
            return np.append(np.zeros((1, n)), theta[1:], axis=0)

    def predict(self, theta, X):
        idx_to_start = 0
        prev_layer = InputLayer(X)
        layers = self.hidden_layers + [self.output_layer]
        for current_layer in layers:
            # reshape weight parameters into 2d array
            prev_layer_size = prev_layer.A.shape[1]
            current_theta_size = prev_layer_size * current_layer.size
            idx_to_end = idx_to_start + current_theta_size
            current_theta_unrolled = theta[idx_to_start:idx_to_end]
            current_layer.theta = current_theta_unrolled.reshape(
                (prev_layer_size, current_layer.size))
            # forwardprop
            current_layer.A, current_layer.Z = current_layer.forwardprop(
                current_layer.theta, prev_layer.A)
            # output layer does not need bias term
            if current_layer.layer_type == 'output':
                break
            # prepare for next layer
            current_layer.A = add_bias(current_layer.A)
            prev_layer = current_layer
            idx_to_start += current_theta_size
        return self.output_layer.A

    def loss_function(self, theta, X, Y):
        # forwardprop
        self.predict(theta, X)
        # calc loss
        m = X.shape[0]
        loss = 0
        for current_layer in self.hidden_layers:
            loss += self.l2_reg * np.sum(self._theta_for_l2reg(current_layer.theta) ** 2) / (2 * m)
        loss += self.output_layer.loss_function(
            self.output_layer.theta, self.hidden_layers[-1].A, Y)
        return loss

    def gradient(self, theta, X, Y):
        # forwardprop
        self.predict(theta, X)
        # backprop
        gradient = np.array([])
        current_layer = self.output_layer
        layers = [InputLayer(X)] + self.hidden_layers
        layers.reverse()
        for prev_layer in layers:
            if current_layer.layer_type == 'output':
                grad, dLdA = current_layer.backprop(
                    current_layer.theta, prev_layer.A, Y,
                    current_layer.Z, current_layer.A)
            else:
                grad, dLdA = current_layer.backprop(
                    current_layer.theta, prev_layer.A, dLdA,
                    current_layer.Z)
            gradient = np.concatenate((grad.flatten(), gradient))
            current_layer = prev_layer
        return gradient


class MultiLayerPerceptronScipy(MultiLayerPerceptron, ScipyOptimizerMixin):

    pass


class MultiLayerPerceptronGD(MultiLayerPerceptron, GDOptimizerMixin):

    pass


class MultiLayerPerceptronSAG(MultiLayerPerceptron, SAGOptimizerMixin):

    pass


class MultiLayerPerceptronSGD(MultiLayerPerceptron, SGDOptimizerMixin):

    pass


class MultiLayerPerceptronMomentumSGD(
        MultiLayerPerceptron, MomentumSGDOptimizerMixin):

    pass


class MultiLayerPerceptronRMSprop(MultiLayerPerceptron, RMSpropOptimizerMixin):

    pass


class MultiLayerPerceptronAdaGrad(MultiLayerPerceptron, AdaGradOptimizerMixin):

    pass


class MultiLayerPerceptronAdaDelta(
        MultiLayerPerceptron, AdaDeltaOptimizerMixin):

    pass


class MultiLayerPerceptronAdam(MultiLayerPerceptron, AdamOptimizerMixin):

    pass
