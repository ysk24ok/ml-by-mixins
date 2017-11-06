from unittest import TestCase

import numpy as np
from nose.tools import assert_almost_equal
from numpy.testing import assert_array_almost_equal

from mlbymxn.base import BaseML
from mlbymxn.loss_functions import (
    SquaredLossMixin,
    LogLossMixin,
    HingeLossMixin
)
from mlbymxn.utils import load_data


class MLWithSquaredLoss(BaseML, SquaredLossMixin):

    pass


class TestSquaredLoss(TestCase):

    def setUp(self):
        self.testee = MLWithSquaredLoss()
        self.X, self.y = load_data('ex1data1.txt')
        self.n = self.X.shape[1]

    def test_predict(self):
        self.testee.initialize_theta(np.array([-3.6303, 1.1664]))
        got = self.testee.predict(self.testee.theta, [[1, 3.5]])
        assert_almost_equal(got[0], 0.4521, places=4)

    def test_loss_function(self):
        # theta is initialized by zero
        self.testee.initialize_theta(np.zeros((self.n,)))
        got = self.testee.loss_function(self.testee.theta, self.X, self.y)
        assert_almost_equal(got, 32.07, places=2)
        # theta is initialized by test value
        self.testee.initialize_theta(np.array([-1, 2]))
        got = self.testee.loss_function(self.testee.theta, self.X, self.y)
        assert_almost_equal(got, 54.24, places=2)

    def test_gradient(self):
        # theta is initialized by zero
        self.testee.initialize_theta(np.zeros((self.n,)))
        got = self.testee.gradient(self.testee.theta, self.X, self.y)
        assert_array_almost_equal(
            got, np.array([-5.8391, -65.3288]), decimal=4)
        # theta is initialized by test value
        self.testee.initialize_theta(np.array([-1 , 2]))
        got = self.testee.gradient(self.testee.theta, self.X, self.y)
        assert_array_almost_equal(
            got, np.array([9.4805, 89.3192]), decimal=4)

    def test_hessian(self):
        # theta is initialized by zero
        self.testee.initialize_theta(np.zeros((self.n,)))
        got = self.testee.hessian(self.testee.theta, self.X)
        assert_array_almost_equal(
            got, np.array([[1, 8.1598],[8.1598, 81.4039]]), decimal=4)


class TestSquaredLossWithL2Reg(TestCase):

    def setUp(self):
        self.testee = MLWithSquaredLoss(l2_reg=1)
        self.X, self.y = load_data('ex1data1.txt')
        self.n = self.X.shape[1]

    def test_loss_function(self):
        # theta is initialized by test value
        self.testee.initialize_theta(np.ones((self.n,)))
        got = self.testee.loss_function(self.testee.theta, self.X, self.y)
        assert_almost_equal(got, 10.2717, places=4)

    def test_gradient(self):
        # theta is initialized by test value
        self.testee.initialize_theta(np.ones((self.n,)))
        got = self.testee.gradient(self.testee.theta, self.X, self.y)
        assert_array_almost_equal(
            got, np.array([3.3207, 24.2452]), decimal=4)


class MLWithLogLoss(BaseML, LogLossMixin):

    pass


class TestLogLoss(TestCase):

    def setUp(self):
        self.testee = MLWithLogLoss()
        self.X, self.y = load_data('ex2data1.txt')
        self.n = self.X.shape[1]

    def test_predict(self):
        self.testee.initialize_theta(
            np.array([-25.1613, 0.2062, 0.2015])
        )
        got = self.testee.predict(self.testee.theta, [[1, 45, 85]])
        assert_almost_equal(got[0], 0.7765, places=4)

    def test_loss_function(self):
        # theta is initialized by zero
        self.testee.initialize_theta(np.zeros((self.n,)))
        got = self.testee.loss_function(self.testee.theta, self.X, self.y)
        assert_almost_equal(got, 0.693, places=3)
        # theta is initialized by test value
        self.testee.initialize_theta(np.array([-24, 0.2, 0.2]))
        got = self.testee.loss_function(self.testee.theta, self.X, self.y)
        assert_almost_equal(got, 0.218, places=3)

    def test_gradient(self):
        # theta is initialized by zero
        self.testee.initialize_theta(np.zeros((self.n,)))
        got = self.testee.gradient(self.testee.theta, self.X, self.y)
        assert_array_almost_equal(
            got, np.array([-0.1, -12.0092, -11.2628]), decimal=4)
        # theta is initialized by test value
        self.testee.initialize_theta(np.array([-24, 0.2, 0.2]))
        got = self.testee.gradient(self.testee.theta, self.X, self.y)
        assert_array_almost_equal(
            got, np.array([0.043, 2.566, 2.647]), decimal=3)

    def test_hessian(self):
        # theta is initialized by zero
        self.testee.initialize_theta(np.zeros((self.n,)))
        got = self.testee.hessian(self.testee.theta, self.X)
        expected = np.array([
            [2.50000000e+01, 1.64110685e+03, 1.65554995e+03],
            [1.64110685e+03, 1.17100173e+05, 1.08465593e+05],
            [1.65554995e+03, 1.08465593e+05, 1.18180491e+05]
        ])
        assert_array_almost_equal(got, expected, decimal=3)


class TestLogLossWithL2Reg(TestCase):

    def setUp(self):
        self.testee = MLWithLogLoss(l2_reg=10)
        self.X, self.y = load_data('ex2data2.txt')
        self.n = self.X.shape[1]

    def test_loss_function(self):
        # theta is initialized by test value
        self.testee.initialize_theta(np.ones((self.n,)))
        got = self.testee.loss_function(self.testee.theta, self.X, self.y)
        assert_almost_equal(got, 1.0238, places=4)

    def test_gradient(self):
        # theta is initialized by test value
        self.testee.initialize_theta(np.ones((self.n,)))
        got = self.testee.gradient(self.testee.theta, self.X, self.y)
        assert_array_almost_equal(
            got, np.array([0.264, 0.1532, 0.1717]), decimal=4)


class MLWithHingeLoss(BaseML, HingeLossMixin):

    def __init__(self, threshold: float):
        self.threshold = threshold


class TestHingeLoss(TestCase):

    def setUp(self):
        self.X = np.array([
            [1, 0.6],
            [1, -1.2],
            [1, 0.3],
        ])
        self.y = np.array([1, 1, -1])
        self.n = self.X.shape[1]

    def test_predict(self):
        testee = MLWithHingeLoss(threshold=0.0)
        testee.initialize_theta(np.ones((self.n,)))
        got = testee.predict(testee.theta, self.X)
        np.testing.assert_array_equal(got, np.array([1, -1, 1]))

    def test_loss_function(self):
        # threshold=0
        testee = MLWithHingeLoss(threshold=0.0)
        testee.initialize_theta(np.ones((self.n,)))
        got = testee.loss_function(testee.theta, self.X, self.y)
        assert_almost_equal(got, 1.5, places=1)
        # threshold=1
        testee = MLWithHingeLoss(threshold=1.0)
        testee.initialize_theta(np.ones((self.n,)))
        got = testee.loss_function(testee.theta, self.X, self.y)
        assert_almost_equal(got, 3.5, places=1)

    def test_gradient(self):
        # threshold=0
        testee = MLWithHingeLoss(threshold=0.0)
        testee.initialize_theta(np.ones((self.n,)))
        got = testee.gradient(testee.theta, self.X, self.y)
        assert_array_almost_equal(got, np.array([1.1, 0.63]))
        # threshold=1
        testee = MLWithHingeLoss(threshold=1.0)
        testee.initialize_theta(np.ones((self.n,)))
        got = testee.gradient(testee.theta, self.X, self.y)
        assert_array_almost_equal(got, np.array([1.1, 2.13]))
