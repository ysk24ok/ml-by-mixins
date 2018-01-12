from ..base import BaseML
from ..activation_functions import ExponentialActivationMixin
from ..loss_functions import PoissonLossMixin
from ..optimizers import (
    ScipyOptimizerMixin,
    GDOptimizerMixin,
    SGDOptimizerMixin,
    SAGOptimizerMixin,
    NewtonOptimizerMixin
)


class PoissonRegression(
        BaseML, PoissonLossMixin, ExponentialActivationMixin):

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
