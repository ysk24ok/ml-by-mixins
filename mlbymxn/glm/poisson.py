from ..base import BaseML
from ..activation_functions import ExponentialActivationFunctionMixin
from ..loss_functions import PoissonLossMixin
from ..optimizers import (
    ScipyOptimizerMixin,
    GradientDescentMixin,
    StochasticGradientDescentMixin,
    StochasticAverageGradientMixin,
    NewtonMixin
)


class PoissonRegression(
        BaseML, PoissonLossMixin, ExponentialActivationFunctionMixin):

    pass


class PoissonRegressionScipy(PoissonRegression, ScipyOptimizerMixin):

    pass


class PoissonRegressionGD(PoissonRegression, GradientDescentMixin):

    pass


class PoissonRegressionSGD(
        PoissonRegression, StochasticGradientDescentMixin):

    pass


class PoissonRegressionSAG(
        PoissonRegression, StochasticAverageGradientMixin):

    pass


class PoissonRegressionNewton(PoissonRegression, NewtonMixin):

    pass
