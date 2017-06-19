from unittest import TestCase

import numpy as np

from mlbymxn.glm import Perceptron
from mlbymxn.utils import load_data


class TestPerceptron(TestCase):

    def setUp(self):
        X, y = load_data('ex2data2')
        self.X = X
        self.y = np.vectorize(lambda x: x if x == 1 else -1)(y)
        self.n = self.X.shape[1]

    def test_fit(self):
        # theta is initialized by random value
        testee = Perceptron(max_iters=100)
        testee.fit(self.X, self.y)
