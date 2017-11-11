from ..base import BaseML
from ..loss_functions import SquaredLossMixin
from ..optimizers import (
    ScipyOptimizerMixin,
    GradientDescentMixin,
    StochasticGradientDescentMixin,
    StochasticAverageGradientMixin,
    NewtonMixin
)


class LinearRegression(BaseML, SquaredLossMixin):

    pass


class LinearRegressionScipy(LinearRegression, ScipyOptimizerMixin):

    pass


class LinearRegressionGD(LinearRegression, GradientDescentMixin):

    pass


class LinearRegressionSGD(LinearRegression, StochasticGradientDescentMixin):

    def __init__(self, shuffle: bool=True, batch_size: int=1, **kargs):
        super().__init__(**kargs)
        # shuffle training samples every iteration
        self.shuffle = shuffle
        # number of training samples to be used in gradient calculation
        # batch_size=1   -> SGD
        # 1<batch_size<m -> mini-batch SGD
        # batch_size=m   -> GD
        self.batch_size = batch_size


class LinearRegressionSAG(LinearRegression, StochasticAverageGradientMixin):

    def __init__(self, shuffle: bool=True, **kargs):
        super().__init__(**kargs)
        # shuffle training samples every iteration
        self.shuffle = shuffle


class LinearRegressionNewton(LinearRegression, NewtonMixin):

    pass
