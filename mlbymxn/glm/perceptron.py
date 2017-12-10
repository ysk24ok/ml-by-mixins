from ..base import BaseML
from ..activation_functions import StepActivationFunctionMixin
from ..loss_functions import HingeLossMixin
from ..optimizers import (
    ScipyOptimizerMixin,
    StochasticGradientDescentMixin
)


class BasePerceptron(BaseML, HingeLossMixin, StepActivationFunctionMixin):

    def __init__(self, **kargs):
        if 'eta' not in kargs:
            kargs['eta'] = 1
        super().__init__(**kargs)
        # threshold for hinge loss
        self.threshold = 0.0


class PerceptronScipy(BasePerceptron, ScipyOptimizerMixin):

    pass


class Perceptron(BasePerceptron, StochasticGradientDescentMixin):

    pass
