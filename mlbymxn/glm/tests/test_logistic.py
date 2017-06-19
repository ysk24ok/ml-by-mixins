from unittest import TestCase

import numpy as np
from nose.tools import assert_almost_equal

from mlbymxn.glm import (
    LogisticRegressionGD,
    LogisticRegressionSGD,
    LogisticRegressionNewton
)
from mlbymxn.utils import load_data


class TestLogisticRegressionGD(TestCase):

    def setUp(self):
        X, y = load_data('ex2data1')
        self.X = X
        self.y = y
        self.n = self.X.shape[1]

    def test_fit(self):
        # theta is initialized by zero
        testee = LogisticRegressionGD(eta=0.001, max_iters=100)
        testee.initialize_theta(np.zeros((self.n, 1)))
        testee.fit(self.X, self.y)
        assert_almost_equal(testee.theta[0][0], -0.0069, places=4)
        assert_almost_equal(testee.theta[1][0], 0.0105, places=4)
        assert_almost_equal(testee.theta[2][0], 0.0005, places=4)
        loss = testee.loss_function(self.X, self.y)
        assert_almost_equal(loss, 0.6293, places=4)
        # theta is initialized by random value
        testee = LogisticRegressionGD(eta=0.001, max_iters=100)
        testee.fit(self.X, self.y)


class TestLogisticRegressionSGD(TestCase):

    def setUp(self):
        X, y = load_data('ex2data1')
        m, n = X.shape
        self.X = X
        self.y = y
        self.m = m
        self.n = n

    def test_fit(self):
        # GD: theta is initialized by zero
        testee = LogisticRegressionSGD(
            eta=0.001, max_iters=100, batch_size=self.m)
        testee.initialize_theta(np.zeros((self.n, 1)))
        testee.fit(self.X, self.y)
        assert_almost_equal(testee.theta[0][0], -0.0069, places=4)
        assert_almost_equal(testee.theta[1][0], 0.0105, places=4)
        assert_almost_equal(testee.theta[2][0], 0.0005, places=4)
        loss = testee.loss_function(self.X, self.y)
        assert_almost_equal(loss, 0.6293, places=4)
        # GD: theta is initialized by random value
        testee = LogisticRegressionSGD(
            eta=0.001, max_iters=100, batch_size=self.m)
        testee.fit(self.X, self.y)

        # SGD: theta is initialized by zero
        testee = LogisticRegressionSGD(eta=0.001, max_iters=100)
        testee.initialize_theta(np.zeros((self.n, 1)))
        testee.fit(self.X, self.y)
        assert_almost_equal(testee.theta[0][0], -0.7419, places=4)
        assert_almost_equal(testee.theta[1][0], 0.0441, places=4)
        assert_almost_equal(testee.theta[2][0], 0.0347, places=4)
        loss = testee.loss_function(self.X, self.y)
        assert_almost_equal(loss, 1.3992, places=4)
        # SGD: theta is initialized by random value
        testee = LogisticRegressionSGD(eta=0.001, max_iters=100)
        testee.initialize_theta(np.zeros((self.n, 1)))

        # mini-batch SGD: theta is initialized by zero
        testee = LogisticRegressionSGD(eta=0.001, max_iters=100, batch_size=5)
        testee.initialize_theta(np.zeros((self.n, 1)))
        testee.fit(self.X, self.y)
        assert_almost_equal(testee.theta[0][0], -0.1409, places=4)
        assert_almost_equal(testee.theta[1][0], 0.041, places=4)
        assert_almost_equal(testee.theta[2][0], 0.0084, places=4)
        loss = testee.loss_function(self.X, self.y)
        assert_almost_equal(loss, 1.0397, places=4)
        # mini-batch SGD: theta is initialized by random value
        testee = LogisticRegressionSGD(eta=0.001, max_iters=100, batch_size=5)
        testee.initialize_theta(np.zeros((self.n, 1)))


class TestLogisticRegressionNewton(TestCase):

    def setUp(self):
        X, y = load_data('ex2data1')
        self.X = X
        self.y = y
        self.n = self.X.shape[1]

    def test_fit(self):
        # theta is initialized by zero
        testee = LogisticRegressionNewton(eta=1, max_iters=100)
        testee.initialize_theta(np.zeros((self.n, 1)))
        testee.fit(self.X, self.y)
        assert_almost_equal(testee.theta[0][0], -4.2206, places=4)
        assert_almost_equal(testee.theta[1][0], 0.0349, places=4)
        assert_almost_equal(testee.theta[2][0], 0.0327, places=4)
        loss = testee.loss_function(self.X, self.y)
        assert_almost_equal(loss, 0.4276, places=4)
        # theta is initialized by random value
        testee = LogisticRegressionNewton(eta=1, max_iters=100)
        testee.fit(self.X, self.y)
