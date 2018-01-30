from unittest import TestCase

import numpy as np
from nose.tools import assert_almost_equal

from mlbymxn.glm import (
    Perceptron,
    PerceptronByScipy,
    PerceptronBySGD,
)
from mlbymxn.utils import load_data


class TestPerceptron(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.X, y = load_data('ex2data2.txt')
        cls.y = np.vectorize(lambda x: x if x == 1 else -1)(y)

    def test_instantiate(self):
        # no arguments
        c = Perceptron()
        assert_almost_equal(c.threshold, 0.0)
        assert_almost_equal(c.eta, 1.0)
        assert_almost_equal(c.l2_reg, 0.0)
        # pass eta
        c = Perceptron(eta=0.1)
        assert_almost_equal(c.threshold, 0.0)
        assert_almost_equal(c.eta, 0.1)
        # pass l2_reg
        c = Perceptron(l2_reg=1)
        assert_almost_equal(c.threshold, 0.0)
        assert_almost_equal(c.l2_reg, 0.0)

    def test_fit_by_scipy(self):
        c = PerceptronByScipy()
        c.fit(self.X, self.y)

    def test_fit_by_sgd(self):
        c = PerceptronBySGD()
        c.fit(self.X, self.y)
