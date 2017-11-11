from unittest import TestCase

import numpy as np
from nose.tools import assert_almost_equal
from numpy.testing import assert_array_almost_equal

from mlbymxn.glm import (
    PoissonRegressionScipy,
    PoissonRegressionGD,
    PoissonRegressionSGD
)
from mlbymxn.utils import load_data


class TestPoissonRegressionScipy(TestCase):

    def setUp(self):
        self.X, self.y = load_data('data3a.csv')
        self.initial_theta = np.zeros((self.X.shape[1],))

    def test_fit(self):
        # theta is initialized by zero
        testee = PoissonRegressionScipy()
        testee.initialize_theta(self.initial_theta)
        testee.fit(self.X, self.y)
        assert_array_almost_equal(
            testee.theta, np.array([1.2631, 0.0801, -0.0320]), decimal=4)
        loss = testee.loss_function(testee.theta, self.X, self.y)
        assert_almost_equal(loss, 2.3529, places=4)
        # theta is initialized by random value
        testee = PoissonRegressionScipy()
        testee.fit(self.X, self.y)


class TestPoissonRegressionGD(TestCase):

    def setUp(self):
        self.X, self.y = load_data('data3a.csv')
        self.initial_theta = np.zeros((self.X.shape[1],))

    def test_fit(self):
        # theta is initialized by zero
        testee = PoissonRegressionGD(eta=0.001, max_iters=100)
        testee.initialize_theta(self.initial_theta)
        testee.fit(self.X, self.y)
        assert_array_almost_equal(
            testee.theta, np.array([0.0286, 0.1982, -0.0043]), decimal=4)
        loss = testee.loss_function(testee.theta, self.X, self.y)
        assert_almost_equal(loss, 2.4129, places=4)
        # theta is initialized by random value
        testee = PoissonRegressionGD(eta=0.01, max_iters=100)
        testee.fit(self.X, self.y)


class TestPoissonRegressionSGD(TestCase):

    def setUp(self):
        self.X, self.y = load_data('data3a.csv')
        self.initial_theta = np.zeros((self.X.shape[1],))

    def test_fit_gd(self):
        # GD: theta is initialized by zero
        m = self.X.shape[0]
        testee = PoissonRegressionSGD(
            eta=0.001, max_iters=100, batch_size=m)
        testee.initialize_theta(self.initial_theta)
        testee.fit(self.X, self.y)
        assert_array_almost_equal(
            testee.theta, np.array([0.02861, 0.1982, -0.0043]), decimal=4)
        loss = testee.loss_function(testee.theta, self.X, self.y)
        assert_almost_equal(loss, 2.4129, places=4)
        # GD: theta is initialized by random value
        testee = PoissonRegressionSGD(
            eta=0.001, max_iters=100, batch_size=m)
        testee.fit(self.X, self.y)

    def test_fit_sgd(self):
        # SGD: theta is initialized by zero
        testee = PoissonRegressionSGD(eta=0.001, max_iters=100)
        testee.initialize_theta(self.initial_theta)
        testee.fit(self.X, self.y)
        # SGD: theta is initialized by random value
        testee = PoissonRegressionSGD(eta=0.001, max_iters=100)
        testee.fit(self.X, self.y)

    def test_fit_sgd_without_shuffling(self):
        # SGD: theta is initialized by zero
        testee = PoissonRegressionSGD(
            eta=0.001, max_iters=100, shuffle=False)
        testee.initialize_theta(self.initial_theta)
        testee.fit(self.X, self.y)
        assert_array_almost_equal(
            testee.theta, np.array([0.5149, 0.1372, 0.2389]), decimal=4)
        loss = testee.loss_function(testee.theta, self.X, self.y)
        assert_almost_equal(loss, 2.4553, places=4)
        # SGD: theta is initialized by random value
        testee = PoissonRegressionSGD(
            eta=0.001, max_iters=100, shuffle=False)
        testee.fit(self.X, self.y)

    def test_fit_minibatch_sgd(self):
        # mini-batch SGD: theta is initialized by zero
        testee = PoissonRegressionSGD(
            eta=0.001, max_iters=100, batch_size=5)
        testee.initialize_theta(self.initial_theta)
        testee.fit(self.X, self.y)
        # mini-batch SGD: theta is initialized by random value
        testee = PoissonRegressionSGD(
            eta=0.001, max_iters=100, batch_size=5)
        testee.fit(self.X, self.y)

    def test_fit_minibatch_sgd_without_shuffling(self):
        # mini-batch SGD: theta is initialized by zero
        testee = PoissonRegressionSGD(
            eta=0.001, max_iters=100, batch_size=5, shuffle=False)
        testee.initialize_theta(self.initial_theta)
        testee.fit(self.X, self.y)
        assert_array_almost_equal(
            testee.theta, np.array([0.1834, 0.1874, -0.1095]), decimal=4)
        loss = testee.loss_function(testee.theta, self.X, self.y)
        assert_almost_equal(loss, 2.3972, places=4)
        # mini-batch SGD: theta is initialized by random value
        testee = PoissonRegressionSGD(
            eta=0.001, max_iters=100, batch_size=5, shuffle=False)
        testee.fit(self.X, self.y)
