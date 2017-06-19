from ..base import BaseML
from ..loss_functions import HingeLossMixin
from ..optimizers import StochasticGradientDescentMixin


class Perceptron(BaseML, HingeLossMixin, StochasticGradientDescentMixin):

    def __init__(self, shuffle: bool=True, **kargs):
        # learning rate is set to 1 by default
        if 'eta' not in kargs:
            kargs['eta'] = 1
        super().__init__(**kargs)
        # threshold for hinge loss
        self.threshold = 0.0
        # shuffle training samples every iteration
        self.shuffle = shuffle
        # number of training samples to be used in gradient calculation
        self.batch_size = 1
