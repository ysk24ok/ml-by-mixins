from unittest import TestCase

import numpy as np
from nose.tools import assert_equal, assert_true
from numpy.testing import assert_array_equal

from mlbymxn import base
from mlbymxn.optimizers import (
    SGDOptimizerMixin,
    AdaGradOptimizerMixin,
    AdaDeltaOptimizerMixin,
    AdamOptimizerMixin,
)


class MLSGD(base.BaseML, SGDOptimizerMixin):

    pass


class MLAdaGrad(base.BaseML, AdaGradOptimizerMixin):

    pass


class MLAdaDelta(base.BaseML, AdaDeltaOptimizerMixin):

    pass


class MLAdam(base.BaseML, AdamOptimizerMixin):

    pass


class TestBaseML(TestCase):

    def test_set_eta(self):
        # no optimizer
        base_ml = base.BaseML(eta=100)
        assert_equal(base_ml.eta, 100)
        base_ml = base.BaseML()
        assert_equal(base_ml.eta, 0.01)
        # adagrad
        ml_adagrad = MLAdaGrad(eta=100)
        assert_equal(ml_adagrad.eta, 100)
        ml_adagrad = MLAdaGrad()
        assert_equal(ml_adagrad.eta, 0.001)
        # adadelta
        ml_adadelta = MLAdaDelta(eta=100)
        assert_equal(ml_adadelta.eta, 100)
        ml_adadelta = MLAdaDelta()
        assert_equal(ml_adadelta.eta, 1)
        # adam
        ml_adam = MLAdam(eta=100)
        assert_equal(ml_adam.eta, 100)
        ml_adam = MLAdam()
        assert_equal(ml_adam.eta, 0.001)
        # none of the above
        ml_sgd = MLSGD(eta=100)
        assert_equal(ml_sgd.eta, 100)
        ml_sgd = MLSGD()
        assert_equal(ml_sgd.eta, 0.01)

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
