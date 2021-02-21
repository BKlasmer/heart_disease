#!/usr/bin/env python3
# -*- coding: latin-1 -*-

import numpy as np
from numpy.testing import assert_array_almost_equal
from heart_disease import RandomForest

""" Tests for RandomForest class
"""


def test_evaluate_fbeta():
    # Setup
    desired = [0.6521739130434783,
                0.6521739130434783,
                0.7894736842105263,
                0.7894736842105263,
                0.7894736842105263,
                0.7894736842105263,
                0.9090909090909091,
                0.7142857142857143,
                0.0,
                0.0]

    rf = RandomForest()

    # Exercise
    actual, _ = rf.evaluate_fbeta(np.array([1,0,1,0,1]), np.array([0.8,0.2,0.6,0.6,0.7]), beta=0.5, x_range=np.linspace(0,1,10))

    # Verify
    assert_array_almost_equal(actual, desired)
    # Cleanup - none necessary