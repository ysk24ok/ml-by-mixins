from unittest import TestCase

import numpy as np
from nose.tools import assert_almost_equal

from mlbymxn.glm.linear import (
    LinearRegressionGD,
    LinearRegressionSGD,
    LinearRegressionNewton
)
from mlbymxn.utils import load_data


class TestLinearRegressionGD(TestCase):

    def setUp(self):
        X, y = load_data('ex1data1')
        self.X = X
        self.y = y
        self.n = self.X.shape[1]

    def test_fit(self):
        # theta is initialized by zero
        testee = LinearRegressionGD(eta=0.01, max_iters=1500)
        testee.initialize_theta(np.zeros((self.n, 1)))
        testee.fit(self.X, self.y)
        assert_almost_equal(testee.theta[0][0], -3.6303, places=4)
        assert_almost_equal(testee.theta[1][0], 1.1664, places=4)
        loss = testee.loss_function(self.X, self.y)
        assert_almost_equal(loss, 4.4834, places=4)
        # theta is initialized by random value
        testee = LinearRegressionGD(eta=0.01, max_iters=100)
        testee.fit(self.X, self.y)


class TestLinearRegressionSGD(TestCase):

    def setUp(self):
        X, y = load_data('ex1data1')
        self.X = X
        self.y = y
        self.n = self.X.shape[1]

    def test_fit(self):
        # theta is initialized by zero
        testee = LinearRegressionSGD(eta=0.01, max_iters=100)
        testee.initialize_theta(np.zeros((self.n, 1)))
        testee.fit(self.X, self.y)
        assert_almost_equal(testee.theta[0][0], -3.8482, places=4)
        assert_almost_equal(testee.theta[1][0], 1.0571, places=4)
        loss = testee.loss_function(self.X, self.y)
        assert_almost_equal(loss, 5.1779, places=4)
        # theta is initialized by random value
        testee = LinearRegressionSGD(eta=0.01, max_iters=100)
        testee.fit(self.X, self.y)


class TestLinearRegressionMiniBatchSGD(TestCase):

    def setUp(self):
        X, y = load_data('ex1data1')
        self.X = X
        self.y = y
        self.n = self.X.shape[1]

    def test_fit(self):
        # theta is initialized by zero
        testee = LinearRegressionSGD(eta=0.01, max_iters=100, batch_size=5)
        testee.initialize_theta(np.zeros((self.n, 1)))
        testee.fit(self.X, self.y)
        assert_almost_equal(testee.theta[0][0], -3.8824, places=4)
        assert_almost_equal(testee.theta[1][0], 1.023, places=4)
        loss = testee.loss_function(self.X, self.y)
        assert_almost_equal(loss, 5.6355, places=4)
        # theta is initialized by random value
        testee = LinearRegressionSGD(eta=0.01, max_iters=100, batch_size=5)
        testee.fit(self.X, self.y)


class TestLinearRegressionNewton(TestCase):

    def setUp(self):
        X, y = load_data('ex1data1')
        self.X = X
        self.y = y
        self.n = self.X.shape[1]

    def test_fit(self):
        # theta is initialized by zero
        testee = LinearRegressionNewton(eta=0.1, max_iters=100)
        testee.initialize_theta(np.zeros((self.n, 1)))
        testee.fit(self.X, self.y)
        assert_almost_equal(testee.theta[0][0], -3.8957, places=4)
        assert_almost_equal(testee.theta[1][0], 1.1930, places=4)
        loss = testee.loss_function(self.X, self.y)
        assert_almost_equal(loss, 4.4770, places=4)
        # theta is initialized by random value
        testee = LinearRegressionNewton(eta=0.1, max_iters=100)
        testee.fit(self.X, self.y)
