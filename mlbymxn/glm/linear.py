from ..base import BaseML
from ..loss_functions import SquaredLossMixin
from ..optimizers import (
    GradientDescentMixin,
    StochasticGradientDescentMixin,
    NewtonMixin
)


class LinearRegression(BaseML, SquaredLossMixin):

    pass


class LinearRegressionGD(LinearRegression, GradientDescentMixin):

    pass


class LinearRegressionSGD(LinearRegression, StochasticGradientDescentMixin):

    def __init__(self, shuffle: bool=True, batch_size: int=1, **kargs):
        # shuffle training samples every iteration
        self.shuffle = shuffle
        super().__init__(**kargs)
        # number of training samples to be used in gradient calculation
        # batch_size=1   -> SGD
        # 1<batch_size<m -> mini-batch SGD
        # batch_size=m   -> GD
        self.batch_size = batch_size


class LinearRegressionNewton(LinearRegression, NewtonMixin):

    pass
