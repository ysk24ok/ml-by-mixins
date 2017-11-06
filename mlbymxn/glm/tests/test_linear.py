from unittest import TestCase

import numpy as np
from nose.tools import assert_almost_equal
from numpy.testing import assert_array_almost_equal

from mlbymxn.glm import (
    LinearRegressionScipy,
    LinearRegressionGD,
    LinearRegressionSGD,
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

    def test_fit(self):
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

        # SGD: theta is initialized by zero
        testee = LinearRegressionSGD(eta=0.01, max_iters=100)
        testee.initialize_theta(self.initial_theta)
        testee.fit(self.X, self.y)
        assert_array_almost_equal(
            testee.theta, np.array([-3.8482, 1.0571]), decimal=4)
        loss = testee.loss_function(testee.theta, self.X, self.y)
        assert_almost_equal(loss, 5.1779, places=4)
        # SGD: theta is initialized by random value
        testee = LinearRegressionSGD(eta=0.01, max_iters=100)
        testee.fit(self.X, self.y)

        # mini-batch SGD: theta is initialized by zero
        testee = LinearRegressionSGD(eta=0.01, max_iters=100, batch_size=5)
        testee.initialize_theta(self.initial_theta)
        testee.fit(self.X, self.y)
        assert_array_almost_equal(
            testee.theta, np.array([-4.1677, 1.0634]), decimal=4)
        loss = testee.loss_function(testee.theta, self.X, self.y)
        assert_almost_equal(loss, 5.4857, places=4)
        # mini-batch SGD: theta is initialized by random value
        testee = LinearRegressionSGD(eta=0.01, max_iters=100, batch_size=5)
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
