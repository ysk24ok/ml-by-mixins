from ..base import BaseML
from ..activation_functions import IdentityActivationMixin
from ..loss_functions import HingeLossMixin
from ..optimizers import (
    ScipyOptimizerMixin,
    SGDOptimizerMixin,
    AdamOptimizerMixin
)


class LinearCSVM(BaseML, HingeLossMixin, IdentityActivationMixin):

    def __init__(self, **kargs):
        C = 1.0
        if 'C' in kargs:
            C = kargs['C']
            del kargs['C']
        if C == 0:
            raise ValueError("'C = 0' is not allowed")
        kargs['l2_reg'] = 1 / C
        super(LinearCSVM, self).__init__(**kargs)
        # threshold for hinge loss
        self.threshold = 1.0


class LinearCSVMbyScipy(LinearCSVM, ScipyOptimizerMixin):

    pass


class LinearCSVMbySGD(LinearCSVM, SGDOptimizerMixin):

    pass


class LinearCSVMbyAdam(LinearCSVM, AdamOptimizerMixin):

    pass
