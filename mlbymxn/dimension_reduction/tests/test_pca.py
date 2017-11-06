from unittest import TestCase

import numpy as np
from nose.tools import assert_tuple_equal, assert_almost_equal
from numpy.testing import assert_array_almost_equal
from sklearn.preprocessing import StandardScaler

from mlbymxn.dimension_reduction import PCA
from mlbymxn.utils import load_data


class TestPCA(TestCase):

    def setUp(self):
        X = load_data('ex7data1.csv')
        scaler = StandardScaler()
        self.X = scaler.fit_transform(X)
        self.pca = PCA(1)

    def test_fit(self):
        self.pca.fit(self.X)
        # U
        expected_U = np.array([[-0.7071, -0.7071], [-0.7071, 0.7071]])
        assert_array_almost_equal(self.pca.U, expected_U, decimal=4)
        # S
        expected_S = np.array([1.7355, 0.2645])
        assert_array_almost_equal(self.pca.S, expected_S, decimal=4)
        # V
        expected_V = np.array([[-0.7071, -0.7071], [-0.7071, 0.7071]])
        assert_array_almost_equal(self.pca.V, expected_V, decimal=4)

    def test_tansform(self):
        m = self.X.shape[0]
        self.pca.fit(self.X)
        got = self.pca.transform(self.X)
        assert_tuple_equal(got.shape, (m, 1))
        assert_almost_equal(got[0][0], 1.4963, places=4)
        assert_almost_equal(got[1][0], -0.9222, places=4)
        #assert_array_almost_equal(got, expected, decimal=4)

    def test_inverse_transform(self):
        m = len(self.X)
        self.pca.fit(self.X)
        transformed_X = self.pca.transform(self.X)
        got = self.pca.inverse_transform(transformed_X)
        assert_tuple_equal(got.shape, (m, 2))
        assert_array_almost_equal(got[0], np.array([-1.0581,-1.0581]), decimal=4)
        assert_array_almost_equal(got[1], np.array([0.6521,0.6521]), decimal=4)
