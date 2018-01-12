from ..base import BaseML
from ..activation_functions import IdentityActivationMixin
from ..loss_functions import SquaredLossMixin
from ..optimizers import (
    ScipyOptimizerMixin,
    GDOptimizerMixin,
    SAGOptimizerMixin,
    SGDOptimizerMixin,
    NewtonOptimizerMixin
)


class LinearRegression(
        BaseML, SquaredLossMixin, IdentityActivationMixin):

    pass


class LinearRegressionScipy(LinearRegression, ScipyOptimizerMixin):

    pass


class LinearRegressionGD(LinearRegression, GDOptimizerMixin):

    pass


class LinearRegressionSGD(LinearRegression, SGDOptimizerMixin):

    pass


class LinearRegressionSAG(LinearRegression, SAGOptimizerMixin):

    pass


class LinearRegressionNewton(LinearRegression, NewtonOptimizerMixin):

    pass
