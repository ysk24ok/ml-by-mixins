from unittest import TestCase

import numpy as np
from nose.tools import assert_almost_equal
from numpy.testing import assert_array_almost_equal
from scipy.optimize import check_grad

from mlbymxn.nn import (
    MultiLayerPerceptron,
    MultiLayerPerceptronGD
)
from mlbymxn.utils import add_bias, load_data


class TestMultiLayerPerceptron(TestCase):

    def setUp(self):
        self.X = add_bias(np.array([
            [-0.31178367, -2.48678065,  1.63929108, -0.33588161,  0.07612761],
            [ 0.72900392,  0.91325152, -0.4298936 ,  1.23773784, -0.15512816],
            [ 0.21782079,  1.12706373,  2.63128056,  0.11112817,  0.63422534],
            [-0.8990918 , -1.51409323,  0.60182225,  0.12915125,  0.810655  ],
        ]))
        theta1 = np.array([
            [ 1.38503523, -0.51962709, -0.78015214,  0.95560959],
            [ 0.35480861, -1.17643148, -0.58661296, -0.02195668],
            [ 1.81259031,  1.56448966, -1.48185327, -2.12714455],
            [-1.3564758 ,  0.71270509,  0.85724762, -0.83440747],
            [-0.46363197, -0.1810066 ,  0.94309899, -0.46550831],
            [ 0.82465384,  0.53419953,  0.11444143,  0.23371059],
        ])
        theta2 = np.array([
            [ 1.50278553, -0.59545972,  0.52834106],
            [-0.12673638, -0.56147088, -0.37550472],
            [-1.36861282, -1.0335199 ,  0.39636757],
            [ 1.21848065,  0.35877096, -0.47144628],
            [-0.85750144,  1.07368134,  2.33660781],
        ])
        theta3 = np.array([
            [-0.16236698],
            [0.9398248  ],
            [0.42628539 ],
            [-0.75815703]
        ])
        self.Y = np.array([[0],[1],[0],[0]])
        self.theta = np.concatenate((theta1.flatten(), theta2.flatten(), theta3.flatten()))

    def test_predict(self):
        testee = MultiLayerPerceptron((4, 3), 1)
        testee.theta = self.theta
        got = testee.predict(testee.theta, self.X)
        expected = np.array([
            [0.55865298],
            [0.52006807],
            [0.51071853],
            [0.53912084]
        ])
        assert_array_almost_equal(got, expected, decimal=5)

    def test_loss_function(self):
        testee = MultiLayerPerceptron((4, 3), 1)
        testee.theta = self.theta
        got = testee.loss_function(testee.theta, self.X, self.Y)
        assert_almost_equal(got, 0.740289031397, places=5)

    def test_gradient(self):
        testee = MultiLayerPerceptron((4, 3), 1)
        testee.theta = self.theta
        got = testee.gradient(testee.theta, self.X, self.Y)
        expected = np.array([
            -5.76718980e-04, -1.54666864e-03,  8.68656246e-03,  7.35430304e-03,
             8.43120288e-04,  1.63962459e-02, -3.03962949e-03,  7.12208035e-03,
             1.68556029e-03,  2.67043299e-02,  4.36468467e-03,  7.74038770e-03,
             1.49750997e-04, -1.90371898e-02,  3.41442220e-02, -9.54916394e-03,
            -3.78988378e-05,  1.34609079e-02, -4.90830859e-03,  1.18560313e-02,
            -5.14206472e-04, -1.03131905e-02,  9.61816993e-03, -3.04617851e-03,
             3.97285399e-02,  2.45855626e-02, -1.84589967e-02, -3.60791923e-03,
            -2.58391750e-03,  4.35309946e-03,  2.37425752e-02,  7.88739825e-03,
            -1.70338833e-02,  4.38542132e-02,  2.75107511e-02, -2.26563394e-02,
             3.12147343e-02,  2.44429363e-02, -1.16699336e-02,  2.82140106e-01,
             2.27314332e-01,  1.69032209e-01,  2.50595314e-01])
        assert_array_almost_equal(got, expected, decimal=5)

    def test_check_gradient(self):
        testee = MultiLayerPerceptron((4, 3), 1)
        testee.theta = self.theta
        got = check_grad(
            testee.loss_function, testee.gradient,
            testee.theta, self.X, self.Y)
        assert_almost_equal(got, 0, places=6)

    def test_loss_function_with_l2reg(self):
        testee = MultiLayerPerceptron((4, 3), 1, l2_reg=0.1)
        testee.theta = self.theta
        got = testee.loss_function(testee.theta, self.X, self.Y)
        assert_almost_equal(got, 1.17614262099, places=5)

    def test_gradient_with_l2reg(self):
        testee = MultiLayerPerceptron((4, 3), 1, l2_reg=0.1)
        testee.theta = self.theta
        got = testee.gradient(testee.theta, self.X, self.Y)
        expected = np.array([
            -5.76718980e-04, -1.54666864e-03,  8.68656246e-03,  7.35430304e-03,
             9.71333554e-03, -1.30145411e-02, -1.77049535e-02,  6.57316335e-03,
             4.70003180e-02,  6.58165714e-02, -3.26816471e-02, -4.54382260e-02,
            -3.37621440e-02, -1.21956255e-03,  5.55754125e-02, -3.04093507e-02,
            -1.16286981e-02,  8.93574285e-03,  1.86691662e-02,  2.18323585e-04,
             2.01021395e-02,  3.04179775e-03,  1.24792057e-02,  2.79658624e-03,
             3.97285399e-02,  2.45855626e-02, -1.84589967e-02, -6.77632873e-03,
            -1.66206895e-02, -5.03451854e-03, -1.04727453e-02, -1.79505993e-02,
            -7.12469409e-03,  7.43162294e-02,  3.64800251e-02, -3.44424964e-02,
             9.77719833e-03,  5.12849698e-02,  4.67452616e-02,  2.82140106e-01,
             2.50809952e-01,  1.79689344e-01,  2.31641388e-01])
        assert_array_almost_equal(got, expected, decimal=5)

    def test_check_gradient_with_l2reg(self):
        testee = MultiLayerPerceptron((4, 3), 1, l2_reg=0.1)
        testee.theta = self.theta
        got = check_grad(
            testee.loss_function, testee.gradient,
            testee.theta, self.X, self.Y)
        assert_almost_equal(got, 0, places=6)
