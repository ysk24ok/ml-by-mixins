from unittest import TestCase

#import numpy as np
from nose.tools import assert_tuple_equal, assert_almost_equal
from sklearn.preprocessing import StandardScaler

from mlbymxn.dimension_reduction import PCA
from mlbymxn.utils import load_data_as_dataframe


class TestPCA(TestCase):

    def setUp(self):
        X = load_data_as_dataframe('ex7data1.csv')
        scaler = StandardScaler()
        self.X = scaler.fit_transform(X)
        self.pca = PCA(1)

    def fit(self):
        self.pca.fit(self.X)
        # U
        assert_almost_equal(self.pca.U[0][0], -0.7071, places=4)
        assert_almost_equal(self.pca.U[0][1], -0.7071, places=4)
        assert_almost_equal(self.pca.U[1][0], -0.7071, places=4)
        assert_almost_equal(self.pca.U[1][1], 0.7071, places=4)
        # S
        assert_almost_equal(self.pca.S[0], 1.7355, places=4)
        assert_almost_equal(self.pca.S[0], 0.2645, places=4)
        # V
        assert_almost_equal(self.pca.V[0][0], -0.7071, places=4)
        assert_almost_equal(self.pca.V[0][1], -0.7071, places=4)
        assert_almost_equal(self.pca.V[1][0], -0.7071, places=4)
        assert_almost_equal(self.pca.V[1][1], 0.7071, places=4)

    def test_tansform(self):
        m = len(self.X)
        self.pca.fit(self.X)
        transformed_X = self.pca.transform(self.X)
        assert_tuple_equal(transformed_X.shape, (m, 1))
        assert_almost_equal(transformed_X[0][0], 1.4963, places=4)
        assert_almost_equal(transformed_X[1][0], -0.9222, places=4)

    def test_inverse_transform(self):
        m = len(self.X)
        self.pca.fit(self.X)
        transformed_X = self.pca.transform(self.X)
        inv_transformed_X = self.pca.inverse_transform(transformed_X)
        assert_tuple_equal(inv_transformed_X.shape, (m, 2))
        assert_almost_equal(inv_transformed_X[0][0], -1.0581, places=4)
        assert_almost_equal(inv_transformed_X[0][1], -1.0581, places=4)
        assert_almost_equal(inv_transformed_X[1][0], 0.6521, places=4)
        assert_almost_equal(inv_transformed_X[1][1], 0.6521, places=4)
