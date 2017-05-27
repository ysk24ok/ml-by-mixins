from unittest import TestCase

import numpy as np
from nose.tools import assert_almost_equal

from ..logistic import (
    LogisticRegressionGD,
    LogisticRegressionSGD,
    LogisticRegressionNewton
)
from ...utils import load_data


class TestLogisticRegressionGD(TestCase):

    def setUp(self):
        self.testee = LogisticRegressionGD(eta=0.001, max_iters=500)
        X, y = load_data('ex2data1')
        self.X = X
        self.y = y
        self.n = self.X.shape[1]

    def test_fit(self):
        # theta is initialized by zero
        self.testee.initialize_theta(np.zeros((self.n, 1)))
        self.testee.fit(self.X, self.y)
        assert_almost_equal(self.testee.theta[0][0], -0.0348, places=4)
        assert_almost_equal(self.testee.theta[1][0], 0.0107, places=4)
        assert_almost_equal(self.testee.theta[2][0], 0.0007, places=4)
        loss = self.testee.loss_function(self.X, self.y)
        assert_almost_equal(loss, 0.6274, places=4)


class TestLogisticRegressionSGD(TestCase):

    def setUp(self):
        self.testee = LogisticRegressionSGD(eta=0.001, max_iters=500)
        X, y = load_data('ex2data1')
        self.X = X
        self.y = y
        self.n = self.X.shape[1]

    def test_fit(self):
        # theta is initialized by zero
        self.testee.initialize_theta(np.zeros((self.n, 1)))
        self.testee.fit(self.X, self.y)
        assert_almost_equal(self.testee.theta[0][0], -3.5318, places=4)
        assert_almost_equal(self.testee.theta[1][0], 0.0674, places=4)
        assert_almost_equal(self.testee.theta[2][0], 0.0497, places=4)
        loss = self.testee.loss_function(self.X, self.y)
        assert_almost_equal(loss, 1.1202, places=4)


class TestLogisticRegressionNewton(TestCase):

    def setUp(self):
        self.testee = LogisticRegressionNewton(eta=1, max_iters=500)
        X, y = load_data('ex2data1')
        self.X = X
        self.y = y
        self.n = self.X.shape[1]

    def test_fit(self):
        # theta is initialized by zero
        self.testee.initialize_theta(np.zeros((self.n, 1)))
        self.testee.fit(self.X, self.y)
        assert_almost_equal(self.testee.theta[0][0], -9.3641, places=4)
        assert_almost_equal(self.testee.theta[1][0], 0.0774, places=4)
        assert_almost_equal(self.testee.theta[2][0], 0.073, places=4)
        loss = self.testee.loss_function(self.X, self.y)
        assert_almost_equal(loss, 0.287, places=4)
