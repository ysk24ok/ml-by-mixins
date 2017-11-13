from ..base import BaseML, OnlineML
from ..loss_functions import LogLossMixin
from ..optimizers import (
    ScipyOptimizerMixin,
    GradientDescentMixin,
    StochasticGradientDescentMixin,
    StochasticAverageGradientMixin,
    NewtonMixin
)


class LogisticRegression(BaseML, LogLossMixin):

    pass


class LogisticRegressionOnline(OnlineML, LogLossMixin):

    pass


class LogisticRegressionScipy(LogisticRegression, ScipyOptimizerMixin):

    pass


class LogisticRegressionGD(LogisticRegression, GradientDescentMixin):

    pass


class LogisticRegressionSGD(
        LogisticRegressionOnline, StochasticGradientDescentMixin):

    def __init__(self, batch_size: int=1, **kargs):
        super().__init__(**kargs)
        # number of training samples to be used in gradient calculation
        # batch_size=1   -> SGD
        # 1<batch_size<m -> mini-batch SGD
        # batch_size=m   -> GD
        self.batch_size = batch_size


class LogisticRegressionSAG(
        LogisticRegressionOnline, StochasticAverageGradientMixin):

    pass


class LogisticRegressionNewton(LogisticRegression, NewtonMixin):

    pass
