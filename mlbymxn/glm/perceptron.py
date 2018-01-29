from ..base import BaseML
from ..activation_functions import IdentityActivationMixin
from ..loss_functions import HingeLossMixin
from ..optimizers import (
    ScipyOptimizerMixin,
    SGDOptimizerMixin
)


class BasePerceptron(BaseML, HingeLossMixin, IdentityActivationMixin):

    def __init__(self, **kargs):
        if 'eta' not in kargs:
            kargs['eta'] = 1
        super().__init__(**kargs)
        # threshold for hinge loss
        self.threshold = 0.0


class PerceptronScipy(BasePerceptron, ScipyOptimizerMixin):

    pass


class Perceptron(BasePerceptron, SGDOptimizerMixin):

    pass
