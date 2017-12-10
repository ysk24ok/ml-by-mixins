from ..base import BaseML
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


class LinearRegressionScipy(LinearRegression, ScipyOptimizerMixin):

    pass


class LinearRegressionGD(LinearRegression, GradientDescentMixin):

    pass


class LinearRegressionSGD(
        LinearRegression, StochasticGradientDescentMixin):

    pass


class LinearRegressionSAG(
        LinearRegression, StochasticAverageGradientMixin):

    pass


class LinearRegressionNewton(LinearRegression, NewtonMixin):

    pass
