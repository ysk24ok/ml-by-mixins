from unittest import TestCase

import numpy as np
from nose.tools import assert_almost_equal
from numpy.testing import assert_array_almost_equal

from mlbymxn.glm import (
    LogisticRegressionScipy,
    LogisticRegressionGD,
    LogisticRegressionSGD,
    LogisticRegressionSAG,
    LogisticRegressionNewton
)
from mlbymxn.utils import load_data


class TestLogisticRegressionScipy(TestCase):

    def setUp(self):
        self.X, self.y = load_data('ex2data1.txt')

    def test_fit(self):
        testee = LogisticRegressionScipy()
        testee.fit(self.X, self.y)
        loss = testee.loss_function(testee.theta, self.X, self.y)
        assert_array_almost_equal(
            testee.theta, np.array([-25.1614, 0.2062, 0.2015]), decimal=4)
        assert_almost_equal(loss, 0.2035, places=4)


class TestLogisticRegressionGD(TestCase):

    def setUp(self):
        self.X, self.y = load_data('ex2data1.txt')

    def test_fit(self):
        testee = LogisticRegressionGD(
            eta=0.001, max_iters=100, initialization_type='zero')
        testee.fit(self.X, self.y)
        loss = testee.loss_function(testee.theta, self.X, self.y)
        assert_array_almost_equal(
            testee.theta, np.array([-0.0069, 0.0105, 0.0005]), decimal=4)
        assert_almost_equal(loss, 0.6293, places=4)


class TestLogisticRegressionSGD(TestCase):

    def setUp(self):
        self.X, self.y = load_data('ex2data1.txt')

    def test_fit_gd(self):
        m = self.X.shape[0]
        testee = LogisticRegressionSGD(
            eta=0.001, max_iters=100, initialization_type='zero', batch_size=m)
        testee.fit(self.X, self.y)
        loss = testee.loss_function(testee.theta, self.X, self.y)
        assert_array_almost_equal(
            testee.theta, np.array([-0.0069, 0.0105, 0.0005]), decimal=4)
        assert_almost_equal(loss, 0.6293, places=4)

    def test_fit_sgd(self):
        testee = LogisticRegressionSGD(
            eta=0.001, max_iters=100, initialization_type='zero')
        testee.fit(self.X, self.y)

    def test_fit_sgd_without_shuffling(self):
        testee = LogisticRegressionSGD(
            eta=0.001, max_iters=100, initialization_type='zero',
            shuffle=False)
        testee.fit(self.X, self.y)
        loss = testee.loss_function(testee.theta, self.X, self.y)
        assert_array_almost_equal(
            testee.theta, np.array([-0.7419, 0.0441, 0.0347]), decimal=4)
        assert_almost_equal(loss, 1.3992, places=4)

    def test_fit_minibatch_sgd(self):
        testee = LogisticRegressionSGD(
            eta=0.001, max_iters=100, batch_size=5)
        testee.fit(self.X, self.y)

    def test_fit_minibatch_sgd_without_shuffling(self):
        testee = LogisticRegressionSGD(
            eta=0.001, max_iters=100, initialization_type='zero', batch_size=5,
            shuffle=False)
        testee.fit(self.X, self.y)
        loss = testee.loss_function(testee.theta, self.X, self.y)
        assert_array_almost_equal(
            testee.theta, np.array([-0.1409, 0.041, 0.0084]), decimal=4)
        assert_almost_equal(loss, 1.0397, places=4)


class TestLogisticRegressionSAG(TestCase):

    def setUp(self):
        self.X, self.y = load_data('ex2data1.txt')

    def test_fit(self):
        # theta is initialized by zero
        testee = LogisticRegressionSAG(
            eta=0.00002, max_iters=100, initialization_type='zero')
        testee.fit(self.X, self.y)
        loss = testee.loss_function(testee.theta, self.X, self.y)
        assert_array_almost_equal(
            testee.theta, np.array([-0.0137, 0.0105, 0.0006]), decimal=4)
        assert_almost_equal(loss, 0.6289, places=4)

    def test_fit_without_shuffling(self):
        testee = LogisticRegressionSAG(
            eta=0.00002, max_iters=100, initialization_type='zero',
            shuffle=False)
        testee.fit(self.X, self.y)
        loss = testee.loss_function(testee.theta, self.X, self.y)
        assert_array_almost_equal(
            testee.theta, np.array([-0.0137, 0.0105, 0.0006]), decimal=4)
        assert_almost_equal(loss, 0.6289, places=4)


class TestLogisticRegressionNewton(TestCase):

    def setUp(self):
        self.X, self.y = load_data('ex2data1.txt')

    def test_fit(self):
        # theta is initialized by zero
        testee = LogisticRegressionNewton(
            eta=1, max_iters=100, initialization_type='zero')
        testee.fit(self.X, self.y)
        loss = testee.loss_function(testee.theta, self.X, self.y)
        assert_array_almost_equal(
            testee.theta, np.array([-4.2206, 0.0349, 0.0327]), decimal=4)
        assert_almost_equal(loss, 0.4276, places=4)
