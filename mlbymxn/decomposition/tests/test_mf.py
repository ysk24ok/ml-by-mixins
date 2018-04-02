from unittest import TestCase

from nose.tools import assert_almost_equal
from numpy.testing import assert_array_almost_equal
from scipy.optimize import check_grad

from mlbymxn.decomposition import (
    MatrixFactorization,
    MatrixFactorizationSGD
)
from mlbymxn.utils import load_data


X, y = load_data('movie_lens_100k')


class TestMatrixFactorizationSGD(TestCase):

    def test_fit(self):
        # TODO: theta is initialized in advance
        # theta is initialized when fit() is called
        self.testee = MatrixFactorizationSGD(10, max_iters=5)
        self.testee.fit(X, y)
