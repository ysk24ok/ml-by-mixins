from unittest import TestCase

import numpy as np
from nose.tools import assert_equal, assert_almost_equal
from numpy.testing import assert_array_equal, assert_array_almost_equal

from mlbymxn.tests import (
    MLWithSquaredLoss,
    MLWithPoissonLoss,
    MLWithLogLoss,
    MLWithHingeLoss
)


class TestIdentityActivationFunction(TestCase):

    def setUp(self):
        self.testee = MLWithSquaredLoss()

    def test_activation_function(self):
        # scalar
        got = self.testee.activation_function(10)
        assert_equal(got, 10)
        # array
        got = self.testee.activation_function(np.array([1,2,3]))
        assert_array_equal(got, np.array([1,2,3]))

    def test_activation_function_gradient(self):
        # scalar
        z = self.testee.activation_function(10)
        got = self.testee.activation_function_gradient(z)
        assert_equal(got, 1)
        # array
        z = self.testee.activation_function(np.array([1,2,3]))
        got = self.testee.activation_function_gradient(z)
        assert_equal(got, 1)
        #assert_array_equal(got, np.array([1,1,1]))


class TestExponentialActivationFunction(TestCase):

    def setUp(self):
        self.testee = MLWithPoissonLoss()

    def test_activation_function(self):
        # scalar
        got = self.testee.activation_function(1)
        assert_almost_equal(got, 2.7183, places=4)
        # array
        got = self.testee.activation_function(np.array([1,2,3]))
        expected = np.array([2.7183, 7.3891, 20.0855])
        assert_array_almost_equal(got, expected, decimal=4)

    def test_activation_function_gradient(self):
        # scalar
        z = self.testee.activation_function(1)
        got = self.testee.activation_function_gradient(z)
        assert_almost_equal(got, 2.7183, places=4)
        # array
        z = self.testee.activation_function(np.array([1,2,3]))
        got = self.testee.activation_function_gradient(z)
        expected = np.array([2.7183, 7.3891, 20.0855])
        assert_array_almost_equal(got, expected, decimal=4)


class TestSigmoidActivationFunction(TestCase):

    def setUp(self):
        self.testee = MLWithLogLoss()

    def test_activation_function(self):
        # scalar
        got = self.testee.activation_function(1)
        assert_almost_equal(got, 0.7311, places=4)
        got = self.testee.activation_function(-1000)
        assert_almost_equal(got, 0, places=4)
        # array
        got = self.testee.activation_function(np.array([1,0,-100,100,-1000]))
        expected = np.array([0.7311, 0.5, 0, 1, 0])
        assert_array_almost_equal(got, expected, decimal=4)

    def test_activation_function_gradient(self):
        # scalar
        z = self.testee.activation_function(1)
        got = self.testee.activation_function_gradient(z)
        assert_almost_equal(got, 0.1966, places=4)
        # array
        z = self.testee.activation_function(np.array([1,2,3]))
        got = self.testee.activation_function_gradient(z)
        expected = np.array([0.1966, 0.105, 0.0452])
        assert_array_almost_equal(got, expected, decimal=4)


class TestStepActivationFunction(TestCase):

    def setUp(self):
        self.testee = MLWithHingeLoss(threshold=0.0)

    def test_activation_function(self):
        # scalar
        got = self.testee.activation_function(10)
        assert_equal(got, 1)
        got = self.testee.activation_function(-10)
        assert_equal(got, 0)
        got = self.testee.activation_function(10, neg_label=-1)
        assert_equal(got, 1)
        got = self.testee.activation_function(-10, neg_label=-1)
        assert_equal(got, -1)
        # array
        got = self.testee.activation_function(np.array([10,-10,0]))
        assert_array_equal(got, np.array([1,0,1]))
        got = self.testee.activation_function(np.array([10,-10,0]), neg_label=-1)
        assert_array_equal(got, np.array([1,-1,1]))

    def test_activation_function_gradient(self):
        # TODO
        pass
