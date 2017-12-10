from ..base import BaseML
from ..activation_functions import ExponentialActivationFunctionMixin
from ..loss_functions import PoissonLossMixin
from ..optimizers import (
    ScipyOptimizerMixin,
    GDOptimizerMixin,
    SGDOptimizerMixin,
    SAGOptimizerMixin,
    NewtonOptimizerMixin
)


class PoissonRegression(
        BaseML, PoissonLossMixin, ExponentialActivationFunctionMixin):

    pass


class PoissonRegressionScipy(PoissonRegression, ScipyOptimizerMixin):

    pass


class PoissonRegressionGD(PoissonRegression, GDOptimizerMixin):

    pass


class PoissonRegressionSGD(PoissonRegression, SGDOptimizerMixin):

    pass


class PoissonRegressionSAG(PoissonRegression, SAGOptimizerMixin):

    pass


class PoissonRegressionNewton(PoissonRegression, NewtonOptimizerMixin):

    pass
