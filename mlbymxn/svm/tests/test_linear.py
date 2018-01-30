from unittest import TestCase

import numpy as np
from nose.tools import assert_almost_equal, assert_raises

from mlbymxn.svm import (
    LinearCSVM,
    LinearCSVMbyScipy,
    LinearCSVMbySGD,
    LinearCSVMbyAdam
)
from mlbymxn.utils import load_data


class TestLinearCSVM(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.X, y = load_data('ex2data2.txt')
        cls.y = np.vectorize(lambda x: x if x == 1 else -1)(y)

    def test_instantiate(self):
        # no arguments
        c = LinearCSVM()
        assert_almost_equal(c.threshold, 1.0)
        assert_almost_equal(c.l2_reg, 1.0)
        # pass C != 0
        c = LinearCSVM(C=1000)
        assert_almost_equal(c.threshold, 1.0)
        assert_almost_equal(c.l2_reg, 0.001)
        c = LinearCSVM(C=0.01)
        assert_almost_equal(c.threshold, 1.0)
        assert_almost_equal(c.l2_reg, 100)
        # pass C == 0
        assert_raises(ValueError, LinearCSVM, C=0)

    def test_fit_by_scipy(self):
        c = LinearCSVMbyScipy()
        c.fit(self.X, self.y)

    def test_fit_by_sgd(self):
        c = LinearCSVMbySGD()
        c.fit(self.X, self.y)

    def test_fit_by_adam(self):
        c = LinearCSVMbyAdam()
        c.fit(self.X, self.y)
