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
        self.testee = LinearRegressionGD(eta=0.01, max_iters=1500)
        X, y = load_data('ex1data1')
        self.X = X
        self.y = y
        self.n = self.X.shape[1]

    def test_fit(self):
        # theta is initialized by zero
        self.testee.initialize_theta(np.zeros((self.n, 1)))
        self.testee.fit(self.X, self.y)
        assert_almost_equal(self.testee.theta[0][0], -3.6303, places=4)
        assert_almost_equal(self.testee.theta[1][0], 1.1664, places=4)
        loss = self.testee.loss_function(self.X, self.y)
        assert_almost_equal(loss, 4.4834, places=4)


class TestLinearRegressionSGD(TestCase):

    def setUp(self):
        self.testee = LinearRegressionSGD(eta=0.01, max_iters=100)
        X, y = load_data('ex1data1')
        self.X = X
        self.y = y
        self.n = self.X.shape[1]

    def test_fit(self):
        # theta is initialized by zero
        self.testee.initialize_theta(np.zeros((self.n, 1)))
        self.testee.fit(self.X, self.y)
        assert_almost_equal(self.testee.theta[0][0], -3.8482, places=4)
        assert_almost_equal(self.testee.theta[1][0], 1.0571, places=4)
        loss = self.testee.loss_function(self.X, self.y)
        assert_almost_equal(loss, 5.1779, places=4)


class TestLinearRegressionMiniBatchSGD(TestCase):

    def setUp(self):
        self.testee = LinearRegressionSGD(eta=0.01, max_iters=100, batch_size=5)
        X, y = load_data('ex1data1')
        self.X = X
        self.y = y
        self.n = self.X.shape[1]

    def test_fit(self):
        # theta is initialized by zero
        self.testee.initialize_theta(np.zeros((self.n, 1)))
        self.testee.fit(self.X, self.y)
        assert_almost_equal(self.testee.theta[0][0], -3.8824, places=4)
        assert_almost_equal(self.testee.theta[1][0], 1.023, places=4)
        loss = self.testee.loss_function(self.X, self.y)
        assert_almost_equal(loss, 5.6355, places=4)


class TestLinearRegressionNewton(TestCase):

    def setUp(self):
        self.testee = LinearRegressionNewton(eta=0.1, max_iters=100)
        X, y = load_data('ex1data1')
        self.X = X
        self.y = y
        self.n = self.X.shape[1]

    def test_fit(self):
        # theta is initialized by zero
        self.testee.initialize_theta(np.zeros((self.n, 1)))
        self.testee.fit(self.X, self.y)
        assert_almost_equal(self.testee.theta[0][0], -3.8957, places=4)
        assert_almost_equal(self.testee.theta[1][0], 1.1930, places=4)
        loss = self.testee.loss_function(self.X, self.y)
        assert_almost_equal(loss, 4.4770, places=4)
