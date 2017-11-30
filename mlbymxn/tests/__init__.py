from mlbymxn.base import BaseML
from mlbymxn.activation_functions import (
    IdentityActivationFunctionMixin,
    ExponentialActivationFunctionMixin,
    SigmoidActivationFunctionMixin,
    StepActivationFunctionMixin,
)
from mlbymxn.loss_functions import (
    SquaredLossMixin,
    LogLossMixin,
    HingeLossMixin,
    PoissonLossMixin
)

class MLWithSquaredLoss(
        BaseML, SquaredLossMixin, IdentityActivationFunctionMixin):

    pass


class MLWithPoissonLoss(
        BaseML, PoissonLossMixin, ExponentialActivationFunctionMixin):

    pass


class MLWithLogLoss(BaseML, LogLossMixin, SigmoidActivationFunctionMixin):

    pass


class MLWithHingeLoss(BaseML, HingeLossMixin, StepActivationFunctionMixin):

    def __init__(self, threshold: float):
        self.threshold = threshold
