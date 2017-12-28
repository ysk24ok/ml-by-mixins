from unittest import TestCase

import numpy as np
from nose.tools import assert_almost_equal
from numpy.testing import assert_array_almost_equal

from mlbymxn.glm import PerceptronScipy, Perceptron
from mlbymxn.utils import load_data


class TestPerceptronScipy(TestCase):

    def setUp(self):
        self.X, y = load_data('ex2data2.txt')
        self.y = np.vectorize(lambda x: x if x == 1 else -1)(y)

    def test_fit(self):
        testee = PerceptronScipy(initialization_type='one')
        testee.fit(self.X, self.y)
        expected_theta = np.array([1.2887e-08, 6.6075e-08, 3.9823e-08])
        assert_array_almost_equal(
            testee.theta * 1e+8, expected_theta * 1e+8, decimal=4)
        loss = testee.loss_function(testee.theta, self.X, self.y)
        expected_loss = 2.2305e-06
        assert_almost_equal(loss * 1e+6, expected_loss * 1e+6, places=4)
        #print(sum(testee.predict(testee.theta, self.X) == self.y))


class TestPerceptron(TestCase):

    def setUp(self):
        self.X, y = load_data('ex2data2.txt')
        self.y = np.vectorize(lambda x: x if x == 1 else -1)(y)

    def test_fit(self):
        testee = Perceptron(max_iters=100, initialization_type='one')
        testee.fit(self.X, self.y)
        expected_theta = np.array([-5.1842e-15, -5.3786e-15, -1.8530e-15])
        assert_array_almost_equal(
            testee.theta, expected_theta, decimal=14)
        loss = testee.loss_function(testee.theta, self.X, self.y)
        expected_loss = 3.2615e-13
        assert_almost_equal(loss, expected_loss, places=12)
