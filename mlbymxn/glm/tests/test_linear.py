from unittest import TestCase

import numpy as np
from nose.tools import assert_almost_equal
from numpy.testing import assert_array_almost_equal

from mlbymxn.glm import (
    LinearRegressionScipy,
    LinearRegressionGD,
    LinearRegressionSGD,
    LinearRegressionSAG,
    LinearRegressionNewton
)
from mlbymxn.utils import load_data


class TestLinearRegressionScipy(TestCase):

    def setUp(self):
        self.X, self.y = load_data('ex1data1.txt')
        self.initial_theta = np.zeros((self.X.shape[1],))

    def test_fit(self):
        # theta is initialized by zero
        testee = LinearRegressionScipy()
        testee.initialize_theta(self.initial_theta)
        testee.fit(self.X, self.y)
        assert_array_almost_equal(
            testee.theta, np.array([-3.8958, 1.1930]), decimal=4)
        loss = testee.loss_function(testee.theta, self.X, self.y)
        assert_almost_equal(loss, 4.4770, places=4)
        # theta is initialized by random value
        testee = LinearRegressionScipy()
        testee.fit(self.X, self.y)


class TestLinearRegressionGD(TestCase):

    def setUp(self):
        self.X, self.y = load_data('ex1data1.txt')
        self.initial_theta = np.zeros((self.X.shape[1],))

    def test_fit(self):
        # theta is initialized by zero
        testee = LinearRegressionGD(eta=0.01, max_iters=1500)
        testee.initialize_theta(self.initial_theta)
        testee.fit(self.X, self.y)
        assert_array_almost_equal(
            testee.theta, np.array([-3.6303, 1.1664]), decimal=4)
        loss = testee.loss_function(testee.theta, self.X, self.y)
        assert_almost_equal(loss, 4.4834, places=4)
        # theta is initialized by random value
        testee = LinearRegressionGD(eta=0.01, max_iters=100)
        testee.fit(self.X, self.y)


class TestLinearRegressionSGD(TestCase):

    def setUp(self):
        self.X, self.y = load_data('ex1data1.txt')
        self.initial_theta = np.zeros((self.X.shape[1],))

    def test_fit_gd(self):
        # GD: theta is initialized by zero
        m = self.X.shape[0]
        testee = LinearRegressionSGD(
            eta=0.01, max_iters=1500, batch_size=m)
        testee.initialize_theta(self.initial_theta)
        testee.fit(self.X, self.y)
        assert_array_almost_equal(
            testee.theta, np.array([-3.6303, 1.1664]), decimal=4)
        loss = testee.loss_function(testee.theta, self.X, self.y)
        assert_almost_equal(loss, 4.4834, places=4)
        # GD: theta is initialized by random value
        testee = LinearRegressionSGD(
            eta=0.01, max_iters=100, batch_size=m)
        testee.fit(self.X, self.y)

    def test_fit_sgd(self):
        # SGD: theta is initialized by zero
        testee = LinearRegressionSGD(eta=0.001, max_iters=100)
        testee.initialize_theta(self.initial_theta)
        testee.fit(self.X, self.y)
        # SGD: theta is initialized by random value
        testee = LinearRegressionSGD(eta=0.001, max_iters=100)
        testee.fit(self.X, self.y)

    def test_fit_sgd_without_shuffling(self):
        # SGD: theta is initialized by zero
        testee = LinearRegressionSGD(
            eta=0.01, max_iters=100, shuffle=False)
        testee.initialize_theta(self.initial_theta)
        testee.fit(self.X, self.y)
        assert_array_almost_equal(
            testee.theta, np.array([-3.8482, 1.0571]), decimal=4)
        loss = testee.loss_function(testee.theta, self.X, self.y)
        assert_almost_equal(loss, 5.1779, places=4)
        # SGD: theta is initialized by random value
        testee = LinearRegressionSGD(
            eta=0.01, max_iters=100, shuffle=False)
        testee.fit(self.X, self.y)

    def test_fit_minibatch_sgd(self):
        # mini-batch SGD: theta is initialized by zero
        testee = LinearRegressionSGD(
            eta=0.001, max_iters=100, batch_size=5)
        testee.initialize_theta(self.initial_theta)
        testee.fit(self.X, self.y)
        # mini-batch SGD: theta is initialized by random value
        testee = LinearRegressionSGD(
            eta=0.001, max_iters=100, batch_size=5)
        testee.fit(self.X, self.y)

    def test_fit_minibatch_sgd_without_shuffling(self):
        # mini-batch SGD: theta is initialized by zero
        testee = LinearRegressionSGD(
            eta=0.01, max_iters=100, batch_size=5, shuffle=False)
        testee.initialize_theta(self.initial_theta)
        testee.fit(self.X, self.y)
        assert_array_almost_equal(
            testee.theta, np.array([-3.8824, 1.0230]), decimal=4)
        loss = testee.loss_function(testee.theta, self.X, self.y)
        assert_almost_equal(loss, 5.6355, places=4)
        # mini-batch SGD: theta is initialized by random value
        testee = LinearRegressionSGD(
            eta=0.01, max_iters=100, batch_size=5, shuffle=False)
        testee.fit(self.X, self.y)


class TestLinearRegressionSAG(TestCase):

    def setUp(self):
        self.X, self.y = load_data('ex1data1.txt')
        self.initial_theta = np.zeros((self.X.shape[1],))

    def test_fit(self):
        # theta is initialized by zero
        testee = LinearRegressionSAG(eta=0.0001, max_iters=100)
        testee.initialize_theta(self.initial_theta)
        testee.fit(self.X, self.y)
        assert_array_almost_equal(
            testee.theta, np.array([-0.55, 0.86]), decimal=2)
        loss = testee.loss_function(testee.theta, self.X, self.y)
        assert_almost_equal(loss, 5.49, places=2)
        # theta is initialized by random value
        testee = LinearRegressionSAG(eta=0.0001, max_iters=100)
        testee.fit(self.X, self.y)

    def test_fit_without_shuffling(self):
        # theta is initialized by zero
        testee = LinearRegressionSAG(
            eta=0.0001, max_iters=100, shuffle=False)
        testee.initialize_theta(self.initial_theta)
        testee.fit(self.X, self.y)
        assert_array_almost_equal(
            testee.theta, np.array([-0.5525, 0.8572]), decimal=4)
        loss = testee.loss_function(testee.theta, self.X, self.y)
        assert_almost_equal(loss, 5.4946, places=4)
        # theta is initialized by random value
        testee = LinearRegressionSAG(
            eta=0.0001, max_iters=100, shuffle=False)
        testee.fit(self.X, self.y)


class TestLinearRegressionNewton(TestCase):

    def setUp(self):
        self.X, self.y = load_data('ex1data1.txt')
        self.initial_theta = np.zeros((self.X.shape[1],))

    def test_fit(self):
        # theta is initialized by zero
        testee = LinearRegressionNewton(eta=0.1, max_iters=100)
        testee.initialize_theta(self.initial_theta)
        testee.fit(self.X, self.y)
        assert_array_almost_equal(
            testee.theta, np.array([-3.8957, 1.1930]), decimal=4)
        loss = testee.loss_function(testee.theta, self.X, self.y)
        assert_almost_equal(loss, 4.4770, places=4)
        # theta is initialized by random value
        testee = LinearRegressionNewton(eta=0.1, max_iters=100)
        testee.fit(self.X, self.y)
