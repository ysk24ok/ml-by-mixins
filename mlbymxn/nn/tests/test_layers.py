from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_almost_equal

from mlbymxn.nn.layers import (
    FullyConnectedLayerIdentity,
    FullyConnectedLayerSigmoid
)


class TestFullyConnectedLayerIdentity(TestCase):

    def test_forwardprop(self):
        A_prev = np.array([
            [1, 1.62434536, -0.52817175, 0.86540763],
            [1, -0.61175641, -1.07296862, -2.3015387]
        ])
        theta = np.array([
            [-0.24937038],
            [1.74481176],
            [-0.7612069],
            [0.3190391]
        ])
        layer = FullyConnectedLayerIdentity(theta.shape[1])
        layer.theta = theta
        A, Z = layer.forwardprop(layer.theta, A_prev)
        expected_A = [[3.26295337], [-1.23429987]]
        assert_array_almost_equal(A, expected_A, decimal=5)
        assert_array_almost_equal(Z, expected_A, decimal=5)

    def test_backprop(self):
        dLdA_next = np.array([[1.62434536], [-0.61175641]])
        A_prev = np.array([
            [1, -0.52817175, 0.86540763, 1.74481176],
            [1, -1.07296862, -2.3015387, -0.7612069]
        ])
        theta = np.array([
            [-2.06014071],
            [0.3190391],
            [-0.24937038],
            [1.46210794]
        ])
        layer = FullyConnectedLayerIdentity(theta.shape[1])
        layer.theta = theta
        expected_grad = np.array([
            [0.50629448],
            [-0.10076895],
            [ 1.40685096],
            [ 1.64992504]
        ])
        expected_dLdA = np.array([
            [ 0.51822968, -0.40506362,  2.37496825],
            [-0.19517421,  0.15255393, -0.8944539 ]
        ])
        # does not use cache
        grad, dLdA = layer.backprop(layer.theta, A_prev, dLdA_next)
        assert_array_almost_equal(grad, expected_grad, decimal=5)
        assert_array_almost_equal(dLdA, expected_dLdA, decimal=5)
        # use cache
        Z_cached = A_prev @ theta
        grad, dLdA = layer.backprop(layer.theta, A_prev, dLdA_next, Z_cached)
        assert_array_almost_equal(grad, expected_grad, decimal=5)
        assert_array_almost_equal(dLdA, expected_dLdA, decimal=5)

    def test_backprop_with_l2reg(self):
        # TODO
        pass


class TestFullyConnectedLayerSigmoid(TestCase):

    def test_forwardprop(self):
        A_prev = np.array([
            [1, -0.41675785, -2.1361961, -1.79343559],
            [1, -0.05626683, 1.64027081, -0.84174737]
        ])
        theta = np.array([
            [-0.90900761],
            [0.50288142],
            [-1.24528809],
            [-1.05795222]
        ])
        layer = FullyConnectedLayerSigmoid(theta.shape[1])
        layer.theta = theta
        A, Z = layer.forwardprop(layer.theta, A_prev)
        expected_A = [[0.96890023], [0.11013289]]
        expected_Z = [[ 3.43896134], [-2.08938436]]
        assert_array_almost_equal(A, expected_A, decimal=5)
        assert_array_almost_equal(Z, expected_Z, decimal=5)

    def test_backprop(self):
        dLdA_next = np.array([
            [1.0408174],
            [-1.41846143],
            [1.24586353],
            [1.04962828]
        ])
        A_prev = np.array([
            [1, 2.2644603 , 6.33722569, 10.37508342],
            [1, 1.09971298, 0.,         0.],
            [1, 0.        , 0.        , 1.63635185],
            [1, 1.54036335, 4.48582383, 8.17870169],
        ])
        theta = np.array([
            [-0.16236698],
            [0.9398248],
            [0.42628539],
            [-0.75815703]
        ])
        layer = FullyConnectedLayerSigmoid(theta.shape[1])
        layer.theta = theta
        expected_grad = np.array([
            [-0.00279212],
            [-0.04069787],
            [ 0.11515566],
            [ 0.27912596]
        ])
        expected_dLdA = np.array([
            [ 0.03685681,  0.0167175 , -0.0297324 ],
            [-0.27725846, -0.12575879,  0.22366451],
            [ 0.18546866,  0.08412481, -0.14961764],
            [ 0.04443658,  0.02015553, -0.03584701]
        ])
        # does not use cache
        grad, dLdA = layer.backprop(layer.theta, A_prev, dLdA_next)
        assert_array_almost_equal(grad, expected_grad, decimal=5)
        assert_array_almost_equal(dLdA, expected_dLdA, decimal=5)
        # use cache
        Z_cached = A_prev @ theta
        grad, dLdA = layer.backprop(layer.theta, A_prev, dLdA_next, Z_cached)
        assert_array_almost_equal(grad, expected_grad, decimal=5)
        assert_array_almost_equal(dLdA, expected_dLdA, decimal=5)

    def test_backprop_with_l2reg(self):
        # TODO
        pass
