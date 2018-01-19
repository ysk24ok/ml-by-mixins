from unittest import TestCase

import numpy as np
from nose.tools import assert_equal, assert_almost_equal
from numpy.testing import assert_array_equal, assert_array_almost_equal

from mlbymxn.tests import (
    MLWithSquaredLoss,
    MLWithPoissonLoss,
    MLWithLogLoss,
    MLWithHingeLoss,
    MLWithTanhActivation,
    MLWithReLUActivation
)


class TestIdentityActivation(TestCase):

    def setUp(self):
        self.testee = MLWithSquaredLoss()

    def test_activation(self):
        # scalar
        got = self.testee.activation(10)
        assert_equal(got, 10)
        # array
        got = self.testee.activation(np.array([1,2,3]))
        assert_array_equal(got, np.array([1,2,3]))

    def test_activation_gradient(self):
        # scalar
        got = self.testee.activation_gradient(10)
        assert_equal(got, 1)
        # array
        got = self.testee.activation_gradient(np.array([1,2,3]))
        assert_equal(got, 1)
        #assert_array_equal(got, np.array([1,1,1]))


class TestExponentialActivation(TestCase):

    def setUp(self):
        self.testee = MLWithPoissonLoss()

    def test_activation(self):
        # scalar
        got = self.testee.activation(1)
        assert_almost_equal(got, 2.7183, places=4)
        # array
        got = self.testee.activation(np.array([1,2,3]))
        expected = np.array([2.7183, 7.3891, 20.0855])
        assert_array_almost_equal(got, expected, decimal=4)

    def test_activation_gradient(self):
        # scalar
        got = self.testee.activation_gradient(1)
        assert_almost_equal(got, 2.7183, places=4)
        # array
        got = self.testee.activation_gradient(np.array([1,2,3]))
        expected = np.array([2.7183, 7.3891, 20.0855])
        assert_array_almost_equal(got, expected, decimal=4)


class TestSigmoidActivation(TestCase):

    def setUp(self):
        self.testee = MLWithLogLoss()

    def test_activation(self):
        got = self.testee.activation(np.array([1,0,-100,100,-1000,1000]))
        expected = np.array([0.7311, 0.5, 0, 1, 0, 1])
        assert_array_almost_equal(got, expected, decimal=4)
        np.log(got)     # confirm no RuntimeWarning
        np.log(1-got)   # confiem no RuntimeWarning

    def test_activation_gradient(self):
        got = self.testee.activation_gradient(np.array([1,2,3]))
        expected = np.array([0.1966, 0.105, 0.0452])
        assert_array_almost_equal(got, expected, decimal=4)


class TestStepActivation(TestCase):

    def setUp(self):
        self.testee = MLWithHingeLoss(threshold=0.0)

    def test_activation(self):
        # scalar
        got = self.testee.activation(10)
        assert_equal(got, 1)
        got = self.testee.activation(-10)
        assert_equal(got, 0)
        got = self.testee.activation(10, neg_label=-1)
        assert_equal(got, 1)
        got = self.testee.activation(-10, neg_label=-1)
        assert_equal(got, -1)
        # array
        got = self.testee.activation(np.array([10,-10,0]))
        assert_array_equal(got, np.array([1,0,1]))
        got = self.testee.activation(np.array([10,-10,0]), neg_label=-1)
        assert_array_equal(got, np.array([1,-1,1]))

    def test_activation_gradient(self):
        # TODO
        pass


class TestTanhActivation(TestCase):

    def setUp(self):
        self.testee = MLWithTanhActivation()

    def test_activation(self):
        got = self.testee.activation(np.array([-1,1,3]))
        expected = np.array([-0.76159418, 0.76159418, 0.99505478])
        assert_array_almost_equal(got, expected, decimal=4)

    def test_activation_gradient(self):
        got = self.testee.activation_gradient(np.array([-1,1,3]))
        expected = np.array([0.4199743, 0.4199743, 0.00986598])
        assert_array_almost_equal(got, expected, decimal=4)


class TestReLUActivation(TestCase):

    def setUp(self):
        self.testee = MLWithReLUActivation()

    def test_activation(self):
        got = self.testee.activation(np.array([-5.0,0.0,5.0]))
        expected = np.array([0.0,0.0,5.0])
        assert_array_equal(got, expected)

    def test_activation_gradient(self):
        Z = np.array([-5.0,0.0,5.0])
        got = self.testee.activation_gradient(Z)
        expected = np.array([0.0,1.0,1.0])
        assert_array_equal(got, expected)
