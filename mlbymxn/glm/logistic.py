from ..base import BaseML
from ..activation_functions import SigmoidActivationMixin
from ..loss_functions import LogLossMixin
from ..optimizers import (
    ScipyOptimizerMixin,
    GDOptimizerMixin,
    SGDOptimizerMixin,
    SAGOptimizerMixin,
    NewtonOptimizerMixin
)


class LogisticRegression(
        BaseML, LogLossMixin, SigmoidActivationMixin):

    pass


class LogisticRegressionScipy(LogisticRegression, ScipyOptimizerMixin):

    pass


class LogisticRegressionGD(LogisticRegression, GDOptimizerMixin):

    pass


class LogisticRegressionSGD(LogisticRegression, SGDOptimizerMixin):

    pass


class LogisticRegressionSAG(
        LogisticRegression, SAGOptimizerMixin):

    pass


class LogisticRegressionNewton(LogisticRegression, NewtonOptimizerMixin):

    pass
