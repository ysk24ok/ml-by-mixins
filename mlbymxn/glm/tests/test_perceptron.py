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
        expected_theta = np.array([-0.4081e-08, 1.6109e-08, -1.1026e-08])
        assert_array_almost_equal(
            testee.theta * 1e+8, expected_theta * 1e+8, decimal=4)
        loss = testee.loss_function(testee.theta, self.X, self.y)
        expected_loss = 0.005321e-06
        assert_almost_equal(loss * 1e+6, expected_loss * 1e+6, places=4)
        #print(sum(testee.predict(testee.theta, self.X) == self.y))


class TestPerceptron(TestCase):

    def setUp(self):
        self.X, y = load_data('ex2data2.txt')
        self.y = np.vectorize(lambda x: x if x == 1 else -1)(y)

    def test_fit(self):
        testee = Perceptron(
            max_iters=100, initialization_type='one', shuffle=False)
        testee.fit(self.X, self.y)
        expected_theta = np.array([-1, -1.1714, 0.0248])
        assert_array_almost_equal(
            testee.theta, expected_theta, decimal=4)
        loss = testee.loss_function(testee.theta, self.X, self.y)
        expected_loss = 0.4993
        assert_almost_equal(loss, expected_loss, places=4)
        #print(sum(testee.predict(testee.theta, self.X) == self.y))
