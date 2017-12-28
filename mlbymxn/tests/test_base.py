from unittest import TestCase

import numpy as np
from nose.tools import assert_equal, assert_true
from numpy.testing import assert_array_equal

from mlbymxn import base


class TestBaseML(TestCase):

    def test_initialize_theta(self):
        n = 100
        # initialization_type: normal
        testee = base.BaseML(initialization_type='normal')
        testee._initialize_theta(n)
        assert_equal(len(testee.theta), 100)
        # initialization_type: uniform
        testee = base.BaseML(initialization_type='uniform')
        testee._initialize_theta(n)
        assert_equal(len(testee.theta), 100)
        assert_true((testee.theta <= 0.5).all())
        assert_true((testee.theta >= -0.5).all())
        # initialization_type: zero
        testee = base.BaseML(initialization_type='zero')
        testee._initialize_theta(n)
        assert_array_equal(testee.theta, np.zeros((100,)))
        # initialization_type: one
        testee = base.BaseML(initialization_type='one')
        testee._initialize_theta(n)
        assert_array_equal(testee.theta, np.ones((100,)))
