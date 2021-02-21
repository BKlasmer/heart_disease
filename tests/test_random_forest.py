#!/usr/bin/env python3
# -*- coding: latin-1 -*-

import pickle
import numpy as np
from numpy.testing import assert_array_almost_equal
from heart_disease import RandomForest, DataLoader

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

def test_directional_change_age():
    # Setup
    with open("data/rf_model.pkl", "rb") as f:
        model = pickle.load(f)

    dataloader = DataLoader()
    dataset = dataloader.dataset
    _, features, _ = dataloader.features_and_labels_to_numpy(dataset)
    test_feature = features[0,:]

    # Exercise
    higher_age = model.predict_proba(test_feature.reshape(1, -1))[0,1]
    test_feature[0] -= 0.1 # Lower the age
    lower_age = model.predict_proba(test_feature.reshape(1, -1))[0,1]

    # Verify
    assert higher_age >= lower_age
    # Cleanup - none necessary

def test_directional_change_vessels():
    # Setup
    with open("data/rf_model.pkl", "rb") as f:
        model = pickle.load(f)

    dataloader = DataLoader()
    dataset = dataloader.dataset
    _, features, _ = dataloader.features_and_labels_to_numpy(dataset)
    test_feature = features[0,:]

    # Exercise
    vessels = model.predict_proba(test_feature.reshape(1, -1))[0,1]
    test_feature[8] += 0.1 # Increase number of major vessels
    more_vessels = model.predict_proba(test_feature.reshape(1, -1))[0,1]

    # Verify
    assert vessels <= more_vessels
    # Cleanup - none necessary