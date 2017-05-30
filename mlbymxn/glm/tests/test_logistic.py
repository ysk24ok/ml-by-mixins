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
        X, y = load_data('ex2data1')
        self.X = X
        self.y = y
        self.n = self.X.shape[1]

    def test_fit(self):
        # theta is initialized by zero
        testee = LogisticRegressionGD(eta=0.001, max_iters=500)
        testee.initialize_theta(np.zeros((self.n, 1)))
        testee.fit(self.X, self.y)
        assert_almost_equal(testee.theta[0][0], -0.0348, places=4)
        assert_almost_equal(testee.theta[1][0], 0.0107, places=4)
        assert_almost_equal(testee.theta[2][0], 0.0007, places=4)
        loss = testee.loss_function(self.X, self.y)
        assert_almost_equal(loss, 0.6274, places=4)
        # theta is initialized by random value
        testee = LogisticRegressionGD(eta=0.001, max_iters=100)
        testee.fit(self.X, self.y)


class TestLogisticRegressionSGD(TestCase):

    def setUp(self):
        X, y = load_data('ex2data1')
        self.X = X
        self.y = y
        self.n = self.X.shape[1]

    def test_fit(self):
        # theta is initialized by zero
        testee = LogisticRegressionSGD(eta=0.001, max_iters=500)
        testee.initialize_theta(np.zeros((self.n, 1)))
        testee.fit(self.X, self.y)
        assert_almost_equal(testee.theta[0][0], -3.5318, places=4)
        assert_almost_equal(testee.theta[1][0], 0.0674, places=4)
        assert_almost_equal(testee.theta[2][0], 0.0497, places=4)
        loss = testee.loss_function(self.X, self.y)
        assert_almost_equal(loss, 1.1202, places=4)
        # theta is initialized by random value
        testee = LogisticRegressionSGD(eta=0.001, max_iters=100)
        testee.initialize_theta(np.zeros((self.n, 1)))

class TestLogisticRegressionNewton(TestCase):

    def setUp(self):
        X, y = load_data('ex2data1')
        self.X = X
        self.y = y
        self.n = self.X.shape[1]

    def test_fit(self):
        # theta is initialized by zero
        testee = LogisticRegressionNewton(eta=1, max_iters=500)
        testee.initialize_theta(np.zeros((self.n, 1)))
        testee.fit(self.X, self.y)
        assert_almost_equal(testee.theta[0][0], -9.3641, places=4)
        assert_almost_equal(testee.theta[1][0], 0.0774, places=4)
        assert_almost_equal(testee.theta[2][0], 0.073, places=4)
        loss = testee.loss_function(self.X, self.y)
        assert_almost_equal(loss, 0.287, places=4)
        # theta is initialized by random value
        testee = LogisticRegressionNewton(eta=1, max_iters=100)
        testee.fit(self.X, self.y)
