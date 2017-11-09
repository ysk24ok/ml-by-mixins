from unittest import TestCase

import numpy as np
from nose.tools import assert_almost_equal
from numpy.testing import assert_array_almost_equal

from mlbymxn.glm import (
    LogisticRegressionScipy,
    LogisticRegressionGD,
    LogisticRegressionSGD,
    LogisticRegressionNewton
)
from mlbymxn.utils import load_data


class TestLogisticRegressionScipy(TestCase):

    def setUp(self):
        self.X, self.y = load_data('ex2data1.txt')
        self.initial_theta = np.zeros((self.X.shape[1],))

    def test_fit(self):
        # theta is initialized by zero, l2_reg=0
        testee = LogisticRegressionScipy()
        testee.initialize_theta(self.initial_theta)
        testee.fit(self.X, self.y)
        assert_array_almost_equal(
            testee.theta, np.array([-25.1614, 0.2062, 0.2015]), decimal=4)
        loss = testee.loss_function(testee.theta, self.X, self.y)
        assert_almost_equal(loss, 0.2035, places=4)
        # theta is initialized by random value, l2_reg=0
        testee = LogisticRegressionScipy()
        testee.fit(self.X, self.y)


class TestLogisticRegressionGD(TestCase):

    def setUp(self):
        self.X, self.y = load_data('ex2data1.txt')
        self.initial_theta = np.zeros((self.X.shape[1],))

    def test_fit(self):
        # theta is initialized by zero
        testee = LogisticRegressionGD(eta=0.001, max_iters=100)
        testee.initialize_theta(self.initial_theta)
        testee.fit(self.X, self.y)
        assert_array_almost_equal(
            testee.theta, np.array([-0.0069, 0.0105, 0.0005]), decimal=4)
        loss = testee.loss_function(testee.theta, self.X, self.y)
        assert_almost_equal(loss, 0.6293, places=4)
        # theta is initialized by random value
        testee = LogisticRegressionGD(eta=0.001, max_iters=100)
        testee.fit(self.X, self.y)


class TestLogisticRegressionSGD(TestCase):

    def setUp(self):
        self.X, self.y = load_data('ex2data1.txt')
        self.initial_theta = np.zeros((self.X.shape[1],))

    def test_fit_gd(self):
        # GD: theta is initialized by zero
        m = self.X.shape[0]
        testee = LogisticRegressionSGD(
            eta=0.001, max_iters=100, batch_size=m)
        testee.initialize_theta(self.initial_theta)
        testee.fit(self.X, self.y)
        assert_array_almost_equal(
            testee.theta, np.array([-0.0069, 0.0105, 0.0005]), decimal=4)
        loss = testee.loss_function(testee.theta, self.X, self.y)
        assert_almost_equal(loss, 0.6293, places=4)
        # GD: theta is initialized by random value
        testee = LogisticRegressionSGD(
            eta=0.001, max_iters=100, batch_size=m)
        testee.fit(self.X, self.y)

    def test_fit_sgd(self):
        # SGD: theta is initialized by zero
        testee = LogisticRegressionSGD(eta=0.001, max_iters=100)
        testee.initialize_theta(self.initial_theta)
        testee.fit(self.X, self.y)
        assert_array_almost_equal(
            testee.theta, np.array([-0.7419, 0.0441, 0.0347]), decimal=4)
        loss = testee.loss_function(testee.theta, self.X, self.y)
        assert_almost_equal(loss, 1.3992, places=4)
        # SGD: theta is initialized by random value
        testee = LogisticRegressionSGD(eta=0.001, max_iters=100)
        testee.fit(self.X, self.y)

    def test_fit_minibatch_sgd(self):
        # mini-batch SGD: theta is initialized by zero
        testee = LogisticRegressionSGD(
            eta=0.001, max_iters=100, batch_size=5)
        testee.initialize_theta(self.initial_theta)
        testee.fit(self.X, self.y)
        assert_array_almost_equal(
            testee.theta, np.array([-0.1409, 0.041, 0.0084]), decimal=4)
        loss = testee.loss_function(testee.theta, self.X, self.y)
        assert_almost_equal(loss, 1.0397, places=4)
        # mini-batch SGD: theta is initialized by random value
        testee = LogisticRegressionSGD(
            eta=0.001, max_iters=100, batch_size=5)
        testee.fit(self.X, self.y)


class TestLogisticRegressionNewton(TestCase):

    def setUp(self):
        self.X, self.y = load_data('ex2data1.txt')
        self.initial_theta = np.zeros((self.X.shape[1],))

    def test_fit(self):
        # theta is initialized by zero
        testee = LogisticRegressionNewton(eta=1, max_iters=100)
        testee.initialize_theta(self.initial_theta)
        testee.fit(self.X, self.y)
        assert_array_almost_equal(
            testee.theta, np.array([-4.2206, 0.0349, 0.0327]), decimal=4)
        loss = testee.loss_function(testee.theta, self.X, self.y)
        assert_almost_equal(loss, 0.4276, places=4)
        # theta is initialized by random value
        testee = LogisticRegressionNewton(eta=1, max_iters=100)
        testee.fit(self.X, self.y)
