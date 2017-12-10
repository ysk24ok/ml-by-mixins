from ..base import BaseML
from ..activation_functions import SigmoidActivationFunctionMixin
from ..loss_functions import LogLossMixin
from ..optimizers import (
    ScipyOptimizerMixin,
    GradientDescentMixin,
    StochasticGradientDescentMixin,
    StochasticAverageGradientMixin,
    NewtonMixin
)


class LogisticRegression(
        BaseML, LogLossMixin, SigmoidActivationFunctionMixin):

    pass


class LogisticRegressionScipy(LogisticRegression, ScipyOptimizerMixin):

    pass


class LogisticRegressionGD(LogisticRegression, GradientDescentMixin):

    pass


class LogisticRegressionSGD(
        LogisticRegression, StochasticGradientDescentMixin):

    pass


class LogisticRegressionSAG(
        LogisticRegression, StochasticAverageGradientMixin):

    pass


class LogisticRegressionNewton(LogisticRegression, NewtonMixin):

    pass
