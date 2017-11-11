from ..base import BaseML
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


class LogisticRegressionScipy(LogisticRegression, ScipyOptimizerMixin):

    pass


class LogisticRegressionGD(LogisticRegression, GradientDescentMixin):

    pass


class LogisticRegressionSGD(
        LogisticRegression, StochasticGradientDescentMixin):

    def __init__(self, shuffle: bool=True, batch_size: int=1, **kargs):
        super().__init__(**kargs)
        # shuffle training samples every iteration
        self.shuffle = shuffle
        # number of training samples to be used in gradient calculation
        # batch_size=1   -> SGD
        # 1<batch_size<m -> mini-batch SGD
        # batch_size=m   -> GD
        self.batch_size = batch_size


class LogisticRegressionSAG(
        LogisticRegression, StochasticAverageGradientMixin):

    def __init__(self, shuffle: bool=True, **kargs):
        super().__init__(**kargs)
        # shuffle training samples every iteration
        self.shuffle = shuffle


class LogisticRegressionNewton(LogisticRegression, NewtonMixin):

    pass
