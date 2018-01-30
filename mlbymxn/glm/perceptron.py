from ..base import BaseML
from ..activation_functions import IdentityActivationMixin
from ..loss_functions import HingeLossMixin
from ..optimizers import (
    ScipyOptimizerMixin,
    SGDOptimizerMixin
)


class Perceptron(BaseML, HingeLossMixin, IdentityActivationMixin):

    def __init__(self, **kargs):
        # Force l2_reg = 0.0
        # because Perceptron has no margin-maximization effect
        kargs['l2_reg'] = 0.0
        if 'eta' not in kargs:
            kargs['eta'] = 1.0
        super(Perceptron, self).__init__(**kargs)
        # threshold for hinge loss
        self.threshold = 0.0


class PerceptronByScipy(Perceptron, ScipyOptimizerMixin):

    pass


class PerceptronBySGD(Perceptron, SGDOptimizerMixin):

    pass
