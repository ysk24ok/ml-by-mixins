from unittest import TestCase

import numpy as np
from nose.tools import assert_almost_equal

from mlbymxn.base import BaseML
from mlbymxn.loss_functions import SquaredLossMixin, LogLossMixin
from mlbymxn.utils import load_data


class MLWithSquaredLoss(BaseML, SquaredLossMixin):

    pass


class TestSquaredLoss(TestCase):

    def setUp(self):
        self.testee = MLWithSquaredLoss()
        X, y = load_data('ex1data1')
        self.X = X
        self.y = y
        self.n = self.X.shape[1]

    def test_predict(self):
        self.testee.initialize_theta(np.array([[-3.6303], [1.1664]]))
        got = self.testee.predict([[1, 3.5]])
        assert_almost_equal(got[0][0], 0.4521, places=4)

    def test_loss_function(self):
        # theta is initialized by zero
        self.testee.initialize_theta(np.zeros((self.n, 1)))
        got = self.testee.loss_function(self.X, self.y)
        assert_almost_equal(got, 32.07, places=2)
        # theta is initialized by test value
        self.testee.initialize_theta(np.array([[-1], [2]]))
        got = self.testee.loss_function(self.X, self.y)
        assert_almost_equal(got, 54.24, places=2)

    def test_gradient(self):
        # theta is initialized by zero
        self.testee.initialize_theta(np.zeros((self.n, 1)))
        got = self.testee.gradient(self.X, self.y)
        assert_almost_equal(got[0][0], -5.8391, places=4)
        assert_almost_equal(got[1][0], -65.3288, places=4)
        # theta is initialized by test value
        self.testee.initialize_theta(np.array([[-1], [2]]))
        got = self.testee.gradient(self.X, self.y)
        assert_almost_equal(got[0][0], 9.4805, places=4)
        assert_almost_equal(got[1][0], 89.3192, places=4)

    def test_hessian(self):
        # theta is initialized by zero
        self.testee.initialize_theta(np.zeros((self.n, 1)))
        got = self.testee.hessian(self.X)
        assert_almost_equal(got[0][0], 1, places=4)
        assert_almost_equal(got[0][1], 8.1598, places=4)
        assert_almost_equal(got[1][0], 8.1598, places=4)
        assert_almost_equal(got[1][1], 81.4039, places=4)


class TestSquaredLossWithL2Reg(TestCase):

    def setUp(self):
        self.testee = MLWithSquaredLoss(l2_reg=1)
        X, y = load_data('ex1data1')
        self.X = X
        self.y = y
        self.n = self.X.shape[1]

    def test_loss_function(self):
        # theta is initialized by test value
        self.testee.initialize_theta(np.ones((self.n, 1)))
        got = self.testee.loss_function(self.X, self.y)
        assert_almost_equal(got, 10.2717, places=4)

    def test_gradient(self):
        # theta is initialized by test value
        self.testee.initialize_theta(np.ones((self.n, 1)))
        got = self.testee.gradient(self.X, self.y)
        assert_almost_equal(got[0][0], 3.3207, places=4)
        assert_almost_equal(got[1][0], 24.2452, places=4)


class MLWithLogLoss(BaseML, LogLossMixin):

    pass


class TestLogLoss(TestCase):

    def setUp(self):
        self.testee = MLWithLogLoss()
        X, y = load_data('ex2data1')
        self.X = X
        self.y = y
        self.n = self.X.shape[1]

    def test_predict(self):
        self.testee.initialize_theta(
            np.array([[-25.1613], [0.2062], [0.2015]])
        )
        got = self.testee.predict([[1, 45, 85]])
        assert_almost_equal(got[0][0], 0.7765, places=4)

    def test_loss_function(self):
        # theta is initialized by zero
        self.testee.initialize_theta(np.zeros((self.n, 1)))
        got = self.testee.loss_function(self.X, self.y)
        assert_almost_equal(got, 0.693, places=3)
        # theta is initialized by test value
        self.testee.initialize_theta(np.array([[-24], [0.2], [0.2]]))
        got = self.testee.loss_function(self.X, self.y)
        assert_almost_equal(got, 0.218, places=3)

    def test_gradient(self):
        # theta is initialized by zero
        self.testee.initialize_theta(np.zeros((self.n, 1)))
        got = self.testee.gradient(self.X, self.y)
        assert_almost_equal(got[0][0], -0.1, places=4)
        assert_almost_equal(got[1][0], -12.0092, places=4)
        assert_almost_equal(got[2][0], -11.2628, places=4)
        # theta is initialized by test value
        self.testee.initialize_theta(np.array([[-24], [0.2], [0.2]]))
        got = self.testee.gradient(self.X, self.y)
        assert_almost_equal(got[0][0], 0.043, places=3)
        assert_almost_equal(got[1][0], 2.566, places=3)
        assert_almost_equal(got[2][0], 2.647, places=3)

    def test_hessian(self):
        # theta is initialized by zero
        self.testee.initialize_theta(np.zeros((self.n, 1)))
        got = self.testee.hessian(self.X)
        assert_almost_equal(got[0][0], 2.50000000e+01, places=3)
        assert_almost_equal(got[0][1], 1.64110685e+03, places=3)
        assert_almost_equal(got[0][2], 1.65554995e+03, places=3)
        assert_almost_equal(got[1][1], 1.17100173e+05, places=3)
        assert_almost_equal(got[1][2], 1.08465593e+05, places=3)
        assert_almost_equal(got[2][2], 1.18180491e+05, places=3)


class TestLogLossWithL2Reg(TestCase):

    def setUp(self):
        self.testee = MLWithLogLoss(l2_reg=10)
        X, y = load_data('ex2data2')
        self.X = X
        self.y = y
        self.n = self.X.shape[1]

    def test_loss_function(self):
        # theta is initialized by test value
        self.testee.initialize_theta(np.ones((self.n, 1)))
        got = self.testee.loss_function(self.X, self.y)
        assert_almost_equal(got, 1.0238, places=4)

    def test_gradient(self):
        # theta is initialized by test value
        self.testee.initialize_theta(np.ones((self.n, 1)))
        got = self.testee.gradient(self.X, self.y)
        assert_almost_equal(got[0][0], 0.264, places=4)
        assert_almost_equal(got[1][0], 0.1532, places=4)
        assert_almost_equal(got[2][0], 0.1717, places=4)
