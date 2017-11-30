from ..base import BaseML, OnlineML
from ..activation_functions import IdentityActivationFunctionMixin
from ..loss_functions import SquaredLossMixin
from ..optimizers import (
    ScipyOptimizerMixin,
    GradientDescentMixin,
    StochasticGradientDescentMixin,
    StochasticAverageGradientMixin,
    NewtonMixin
)


class LinearRegression(
        BaseML, SquaredLossMixin, IdentityActivationFunctionMixin):

    pass


class LinearRegressionOnline(
        OnlineML, SquaredLossMixin, IdentityActivationFunctionMixin):

    pass


class LinearRegressionScipy(LinearRegression, ScipyOptimizerMixin):

    pass


class LinearRegressionGD(LinearRegression, GradientDescentMixin):

    pass


class LinearRegressionSGD(
        LinearRegressionOnline, StochasticGradientDescentMixin):

    def __init__(self, batch_size: int=1, **kargs):
        super().__init__(**kargs)
        # number of training samples to be used in gradient calculation
        # batch_size=1   -> SGD
        # 1<batch_size<m -> mini-batch SGD
        # batch_size=m   -> GD
        self.batch_size = batch_size


class LinearRegressionSAG(
        LinearRegressionOnline, StochasticAverageGradientMixin):

    pass


class LinearRegressionNewton(LinearRegression, NewtonMixin):

    pass
