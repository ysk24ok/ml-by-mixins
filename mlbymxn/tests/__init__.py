from mlbymxn.base import BaseML
from mlbymxn.activation_functions import (
    IdentityActivationMixin,
    ExponentialActivationMixin,
    SigmoidActivationMixin,
    StepActivationMixin,
    TanhActivationMixin,
    ReLUActivationMixin,
)
from mlbymxn.loss_functions import (
    SquaredLossMixin,
    LogLossMixin,
    HingeLossMixin,
    PoissonLossMixin
)

class MLWithSquaredLoss(
        BaseML, SquaredLossMixin, IdentityActivationMixin):

    pass


class MLWithPoissonLoss(
        BaseML, PoissonLossMixin, ExponentialActivationMixin):

    pass


class MLWithLogLoss(BaseML, LogLossMixin, SigmoidActivationMixin):

    pass


class MLWithHingeLoss(BaseML, HingeLossMixin, StepActivationMixin):

    def __init__(self, threshold: float):
        self.threshold = threshold


class MLWithTanhActivation(BaseML, TanhActivationMixin):

    pass


class MLWithReLUActivation(BaseML, ReLUActivationMixin):

    pass
