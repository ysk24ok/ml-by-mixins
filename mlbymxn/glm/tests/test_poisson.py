from unittest import TestCase

import numpy as np
from nose.tools import assert_almost_equal
from numpy.testing import assert_array_almost_equal

from mlbymxn.glm import (
    PoissonRegressionScipy,
    PoissonRegressionGD,
    PoissonRegressionSGD,
    PoissonRegressionSAG
)
from mlbymxn.utils import load_data


class TestPoissonRegressionScipy(TestCase):

    def setUp(self):
        self.X, self.y = load_data('data3a.csv')

    def test_fit(self):
        testee = PoissonRegressionScipy()
        testee.fit(self.X, self.y)
        loss = testee.loss_function(testee.theta, self.X, self.y)
        assert_array_almost_equal(
            testee.theta, np.array([1.2631, 0.0801, -0.0320]), decimal=4)
        assert_almost_equal(loss, 2.3529, places=4)


class TestPoissonRegressionGD(TestCase):

    def setUp(self):
        self.X, self.y = load_data('data3a.csv')

    def test_fit(self):
        # theta is initialized by zero
        testee = PoissonRegressionGD(
            eta=0.001, max_iters=100, initialization_type='zero')
        testee.fit(self.X, self.y)
        loss = testee.loss_function(testee.theta, self.X, self.y)
        assert_array_almost_equal(
            testee.theta, np.array([0.0286, 0.1982, -0.0043]), decimal=4)
        assert_almost_equal(loss, 2.4129, places=4)


class TestPoissonRegressionSGD(TestCase):

    def setUp(self):
        self.X, self.y = load_data('data3a.csv')

    def test_fit_gd(self):
        m = self.X.shape[0]
        testee = PoissonRegressionSGD(
            eta=0.001, max_iters=100, initialization_type='zero', batch_size=m)
        testee.fit(self.X, self.y)
        loss = testee.loss_function(testee.theta, self.X, self.y)
        assert_array_almost_equal(
            testee.theta, np.array([0.02861, 0.1982, -0.0043]), decimal=4)
        assert_almost_equal(loss, 2.4129, places=4)

    def test_fit_sgd(self):
        testee = PoissonRegressionSGD(eta=0.001, max_iters=100)
        testee.fit(self.X, self.y)

    def test_fit_sgd_without_shuffling(self):
        testee = PoissonRegressionSGD(
            eta=0.001, max_iters=100, initialization_type='zero',
            shuffle=False)
        testee.fit(self.X, self.y)
        loss = testee.loss_function(testee.theta, self.X, self.y)
        assert_array_almost_equal(
            testee.theta, np.array([0.5149, 0.1372, 0.2389]), decimal=4)
        assert_almost_equal(loss, 2.4553, places=4)

    def test_fit_minibatch_sgd(self):
        testee = PoissonRegressionSGD(
            eta=0.001, max_iters=100, initialization_type='zero', batch_size=5)
        testee.fit(self.X, self.y)

    def test_fit_minibatch_sgd_without_shuffling(self):
        testee = PoissonRegressionSGD(
            eta=0.001, max_iters=100, initialization_type='zero', batch_size=5,
            shuffle=False)
        testee.fit(self.X, self.y)
        loss = testee.loss_function(testee.theta, self.X, self.y)
        assert_array_almost_equal(
            testee.theta, np.array([0.1834, 0.1874, -0.1095]), decimal=4)
        assert_almost_equal(loss, 2.3972, places=4)


class TestPoissonRegressionSAG(TestCase):

    def setUp(self):
        self.X, self.y = load_data('data3a.csv')

    def test_fit(self):
        testee = PoissonRegressionSAG(
            eta=0.00005, max_iters=100, initialization_type='zero')
        testee.fit(self.X, self.y)
        loss = testee.loss_function(testee.theta, self.X, self.y)
        assert_array_almost_equal(
            testee.theta, np.array([0.064, 0.197, -0.042]), decimal=3)
        assert_almost_equal(loss, 2.406, places=4)

    def test_fit_without_shuffling(self):
        testee = PoissonRegressionSAG(
            eta=0.00005, max_iters=100, initialization_type='zero',
            shuffle=False)
        testee.fit(self.X, self.y)
        loss = testee.loss_function(testee.theta, self.X, self.y)
        assert_array_almost_equal(
            testee.theta, np.array([0.06463, 0.1967, -0.0412]), decimal=4)
        assert_almost_equal(loss, 2.4060, places=4)
