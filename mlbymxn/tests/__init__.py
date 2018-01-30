from mlbymxn.base import BaseML
from mlbymxn.activation_functions import (
    IdentityActivationMixin,
    ExponentialActivationMixin,
    SigmoidActivationMixin,
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


class MLWithHingeLoss(BaseML, HingeLossMixin, IdentityActivationMixin):

    def __init__(self, threshold: float, l2_reg: float=0.0):
        self.threshold = threshold
        self.l2_reg = l2_reg


class MLWithTanhActivation(BaseML, TanhActivationMixin):

    pass


class MLWithReLUActivation(BaseML, ReLUActivationMixin):

    pass
