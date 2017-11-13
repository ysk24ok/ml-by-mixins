from ..base import BaseML, OnlineML
from ..loss_functions import PoissonLossMixin
from ..optimizers import (
    ScipyOptimizerMixin,
    GradientDescentMixin,
    StochasticGradientDescentMixin,
    StochasticAverageGradientMixin,
    NewtonMixin
)


class PoissonRegression(BaseML, PoissonLossMixin):

    pass


class PoissonRegressionOnline(OnlineML, PoissonLossMixin):

    pass


class PoissonRegressionScipy(PoissonRegression, ScipyOptimizerMixin):

    pass


class PoissonRegressionGD(PoissonRegression, GradientDescentMixin):

    pass


class PoissonRegressionSGD(
        PoissonRegressionOnline, StochasticGradientDescentMixin):

    def __init__(self, batch_size: int=1, **kargs):
        super().__init__(**kargs)
        # number of training samples to be used in gradient calculation
        # batch_size=1   -> SGD
        # 1<batch_size<m -> mini-batch SGD
        # batch_size=m   -> GD
        self.batch_size = batch_size


class PoissonRegressionSAG(
        PoissonRegressionOnline, StochasticAverageGradientMixin):

    pass


class PoissonRegressionNewton(PoissonRegression, NewtonMixin):

    pass
