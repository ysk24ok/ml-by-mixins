from ..base import OnlineML
from ..activation_functions import StepActivationFunctionMixin
from ..loss_functions import HingeLossMixin
from ..optimizers import (
    ScipyOptimizerMixin,
    StochasticGradientDescentMixin
)


class BasePerceptron(OnlineML, HingeLossMixin, StepActivationFunctionMixin):

    def __init__(self, **kargs):
        # learning rate is set to 1 by default
        if 'eta' not in kargs:
            kargs['eta'] = 1
        super().__init__(**kargs)
        # threshold for hinge loss
        self.threshold = 0.0
        # number of training samples to be used in gradient calculation
        self.batch_size = 1


class PerceptronScipy(BasePerceptron, ScipyOptimizerMixin):

    pass


class Perceptron(BasePerceptron, StochasticGradientDescentMixin):

    pass
