from unittest import TestCase

import numpy as np
from nose.tools import assert_almost_equal
from numpy.testing import assert_array_almost_equal
from scipy.optimize import check_grad

from mlbymxn.tests import (
    MLWithSquaredLoss,
    MLWithLogLoss,
    MLWithPoissonLoss,
    MLWithHingeLoss
)
from mlbymxn.utils import load_data


class TestSquaredLoss(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.testee = MLWithSquaredLoss()
        cls.X, cls.y = load_data('ex1data1.txt')
        cls.n = cls.X.shape[1]

    def test_predict(self):
        theta = np.array([-3.6303, 1.1664])
        got = self.testee.predict(theta, [[1, 3.5]])
        assert_array_almost_equal(got, [0.4521], decimal=4)

    def test_loss_function(self):
        theta = np.array([-1, 2])
        got = self.testee.loss_function(theta, self.X, self.y)
        assert_almost_equal(got, 54.2425, places=4)

    def test_gradient(self):
        theta = np.array([-1, 2])
        got = self.testee.gradient(theta, self.X, self.y)
        assert_array_almost_equal(got, [9.4805, 89.3192], decimal=4)

    def test_hessian(self):
        theta = np.zeros((self.n,))
        got = self.testee.hessian(theta, self.X)
        expected = [[1, 8.1598],[8.1598, 81.4039]]
        assert_array_almost_equal(got, expected, decimal=4)

    def test_check_gradient(self):
        theta = np.random.randn(self.n)
        got = check_grad(
            self.testee.loss_function, self.testee.gradient,
            theta, self.X, self.y)
        assert_almost_equal(got, 0, places=5)


class TestSquaredLossNaiveImpl(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.testee = MLWithSquaredLoss(use_naive_impl=True)
        cls.X, cls.y = load_data('ex1data1.txt')
        cls.n = cls.X.shape[1]

    def test_gradient(self):
        theta = np.array([-1, 2])
        got = self.testee.gradient(theta, self.X, self.y)
        assert_array_almost_equal(got, [9.4805, 89.3192], decimal=4)

    def test_check_gradient(self):
        theta = np.random.randn(self.n)
        got = check_grad(
            self.testee.loss_function, self.testee.gradient,
            theta, self.X, self.y)
        assert_almost_equal(got, 0, places=5)


class TestSquaredLossWithL2Reg(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.testee = MLWithSquaredLoss(l2_reg=1)
        cls.X, cls.y = load_data('ex1data1.txt')
        cls.n = cls.X.shape[1]

    def test_loss_function(self):
        theta = np.array([-1, 2])
        got = self.testee.loss_function(theta, self.X, self.y)
        assert_almost_equal(got, 54.2631, places=4)

    def test_gradient(self):
        theta = np.array([-1, 2])
        got = self.testee.gradient(theta, self.X, self.y)
        assert_array_almost_equal(got, [9.4805, 89.3399], decimal=4)

    def test_check_gradient(self):
        theta = np.random.randn(self.n)
        got = check_grad(
            self.testee.loss_function, self.testee.gradient,
            theta, self.X, self.y)
        assert_almost_equal(got, 0, places=5)


class TestLogLoss(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.testee = MLWithLogLoss()
        cls.X, cls.y = load_data('ex2data1.txt')
        cls.n = cls.X.shape[1]

    def test_predict(self):
        theta = np.array([-25.1613, 0.2062, 0.2015])
        got = self.testee.predict(theta, [[1, 45, 85]])
        assert_array_almost_equal(got, [0.7765], decimal=4)

    def test_loss_function(self):
        theta = np.array([-24, 0.2, 0.2])
        got = self.testee.loss_function(theta, self.X, self.y)
        assert_almost_equal(got, 0.2183, places=4)

    def test_gradient(self):
        theta = np.array([-24, 0.2, 0.2])
        got = self.testee.gradient(theta, self.X, self.y)
        assert_array_almost_equal(got, [0.0429, 2.5662, 2.6468], decimal=4)

    def test_hessian(self):
        theta = np.zeros((self.n,))
        got = self.testee.hessian(theta, self.X)
        expected = [
            [2.50000000e+01, 1.64110685e+03, 1.65554995e+03],
            [1.64110685e+03, 1.17100173e+05, 1.08465593e+05],
            [1.65554995e+03, 1.08465593e+05, 1.18180491e+05]
        ]
        assert_array_almost_equal(got, expected, decimal=3)

    def test_check_gradient(self):
        pass
        # XXX: Sometimes fails when theta is randomly initialized
        #theta = np.random.randn(self.n)
        #got = check_grad(
        #    self.testee.loss_function, self.testee.gradient,
        #    theta, self.X, self.y)
        #assert_almost_equal(got, 0, places=4)


class TestLogLossNaiveImpl(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.testee = MLWithLogLoss(use_naive_impl=True)
        cls.X, cls.y = load_data('ex2data1.txt')
        cls.n = cls.X.shape[1]

    def test_gradient(self):
        theta = np.array([-24, 0.2, 0.2])
        got = self.testee.gradient(theta, self.X, self.y)
        assert_array_almost_equal(got, [0.0429, 2.5662, 2.6468], decimal=4)

    def test_check_gradient(self):
        pass
        # XXX: Sometimes fails when theta is randomly initialized
        #theta = np.random.randn(self.n)
        #got = check_grad(
        #    self.testee.loss_function, self.testee.gradient,
        #    theta, self.X, self.y)
        #assert_almost_equal(got, 0, places=4)


class TestLogLossWithL2Reg(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.testee = MLWithLogLoss(l2_reg=10)
        cls.X, cls.y = load_data('ex2data1.txt')
        cls.n = cls.X.shape[1]

    def test_loss_function(self):
        theta = np.array([-24, 0.2, 0.2])
        got = self.testee.loss_function(theta, self.X, self.y)
        assert_almost_equal(got, 0.2223, places=4)

    def test_gradient(self):
        theta = np.array([-24, 0.2, 0.2])
        got = self.testee.gradient(theta, self.X, self.y)
        assert_array_almost_equal(got, [0.0429, 2.5862, 2.6668], decimal=4)

    def test_check_gradient(self):
        pass
        # XXX: Sometimes fails when theta is randomly initialized
        #theta = np.random.randn(self.n)
        #got = check_grad(
        #    self.testee.loss_function, self.testee.gradient,
        #    theta, self.X, self.y)
        #assert_almost_equal(got, 0, places=4)


class TestHingeLoss(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.X = np.array([
            [1, 0.6],
            [1, -1.2],
            [1, 0.3],
        ])
        cls.y = np.array([1, -1, -1])
        cls.n = cls.X.shape[1]

    def test_predict(self):
        testee = MLWithHingeLoss(threshold=0.0)
        theta = np.ones((self.n,))
        got = testee.predict(theta, self.X)
        np.testing.assert_array_equal(got, np.array([1, -1, 1]))

    def test_loss_function(self):
        # threshold=0
        testee = MLWithHingeLoss(threshold=0.0)
        theta = np.ones((self.n,))
        got = testee.loss_function(theta, self.X, self.y)
        assert_almost_equal(got, 0.4333, places=4)
        # threshold=1
        testee = MLWithHingeLoss(threshold=1.0)
        theta = np.ones((self.n,))
        got = testee.loss_function(theta, self.X, self.y)
        assert_almost_equal(got, 1.0333, places=4)

    def test_gradient(self):
        # threshold=0
        testee = MLWithHingeLoss(threshold=0.0)
        theta = np.array([0.5, 1])
        got = testee.gradient(theta, self.X, self.y)
        assert_array_almost_equal(got, [0.3333, 0.1], decimal=4)
        # threshold=1
        testee = MLWithHingeLoss(threshold=1.0)
        theta = np.array([0.5, 1])
        got = testee.gradient(theta, self.X, self.y)
        assert_array_almost_equal(got, [0.6667, -0.3], decimal=4)

    def test_check_gradient(self):
        pass
        # threshold=0
        testee = MLWithHingeLoss(threshold=0.0)
        theta = np.random.randn(self.n)
        got = check_grad(
            testee.loss_function, testee.gradient, theta, self.X, self.y)
        assert_almost_equal(got, 0, places=4)
        # threshold=1
        testee = MLWithHingeLoss(threshold=1.0)
        theta = np.random.randn(self.n)
        got = check_grad(
            testee.loss_function, testee.gradient, theta, self.X, self.y)
        assert_almost_equal(got, 0, places=4)


class TestPoissonLoss(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.testee = MLWithPoissonLoss()
        cls.X, cls.y = load_data('data3a.csv')
        cls.n = cls.X.shape[1]

    def test_loss_function(self):
        theta = -np.ones((self.n,))
        got = self.testee.loss_function(theta, self.X, self.y)
        assert_almost_equal(got, 102.0244, places=4)

    def test_gradient(self):
        theta = -np.ones((self.n,))
        got = self.testee.gradient(theta, self.X, self.y)
        assert_array_almost_equal(got, [-7.83, -79.5939, -3.94], decimal=4)

    def test_check_gradient(self):
        theta = np.random.randn(self.n)
        got = check_grad(
            self.testee.loss_function, self.testee.gradient,
            theta, self.X, self.y)
        assert_almost_equal(got, 0, places=3)


class TestPoissonLossNaiveImpl(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.testee = MLWithPoissonLoss(use_naive_impl=True)
        cls.X, cls.y = load_data('data3a.csv')
        cls.n = cls.X.shape[1]

    def test_gradient(self):
        theta = -np.ones((self.n,))
        got = self.testee.gradient(theta, self.X, self.y)
        assert_array_almost_equal(got, [-7.83, -79.5939, -3.94], decimal=4)

    def test_check_gradient(self):
        theta = np.random.randn(self.n)
        got = check_grad(
            self.testee.loss_function, self.testee.gradient,
            theta, self.X, self.y)
        assert_almost_equal(got, 0, places=3)


class TestPoissonLossWithL2Reg(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.testee = MLWithPoissonLoss(l2_reg=10)
        cls.X, cls.y = load_data('data3a.csv')
        cls.n = cls.X.shape[1]

    def test_loss_function(self):
        theta = -np.ones((self.n,))
        got = self.testee.loss_function(theta, self.X, self.y)
        assert_almost_equal(got, 102.1244, places=4)

    def test_gradient(self):
        theta = -np.ones((self.n,))
        got = self.testee.gradient(theta, self.X, self.y)
        assert_array_almost_equal(got, [-7.83, -79.6939, -4.04], decimal=4)

    def test_check_gradient(self):
        theta = np.random.randn(self.n)
        got = check_grad(
            self.testee.loss_function, self.testee.gradient,
            theta, self.X, self.y)
        assert_almost_equal(got, 0, places=3)
