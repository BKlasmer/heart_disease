#!/usr/bin/env python3
# -*- coding: latin-1 -*-

import numpy as np
from heart_disease import DataLoader

""" Tests for DataLoader class
"""


def test_ingest_data():
    # Setup
    desired_rows = 303
    desired_columns = 14

    Loader = DataLoader()

    # Exercise
    dataset = Loader._ingest_data()

    # Verify
    assert len(dataset) == desired_rows
    assert len(dataset.columns) == desired_columns
    # Cleanup - none necessary


def test_missing_data():
    # Setup
    desired = 0
    Loader = DataLoader()
    dataset = Loader._ingest_data()

    # Exercise
    dataset = Loader._handle_missing_data(dataset)

    actual = 0
    for column in dataset.columns:
        actual += len(dataset.loc[dataset[column] == "?", column])

    # Verify
    assert desired == actual

    # Cleanup - none necessary


def test_handle_categorical():
    # Setup
    desired = 0
    Loader = DataLoader()
    columns = [
        "Chest Pain Typical",
        "Chest Pain Atypical",
        "Chest Pain Non-anginal",
        "Chest Pain Asymptomatic",
        "Resting ECG Normal",
        "Resting ECG Abnormal",
        "Resting ECG Hypertrophy",
        "Peak Exercise Slope Up",
        "Peak Exercise Slope Flat",
        "Peak Exercise Slope Down",
        "Thal Normal",
        "Thal Fixed Defect",
        "Thal Reversable Defect",
    ]

    dataset = Loader._ingest_data()
    dataset = Loader._handle_missing_data(dataset)

    # Exercise
    dataset = Loader._handle_categorical(dataset)

    actual = 0
    for column in columns:
        actual += len(dataset.loc[dataset[column] > 1, column])

    # Verify
    assert desired == actual

    # Cleanup - none necessary


def test_minmax():
    # Setup
    desired = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

    # Exercise
    actual = DataLoader._minmax(np.array([1, 2, 3, 4, 5]))

    # Verify
    assert all([a == b for a, b in zip(actual, desired)])

    # Cleanup - none necessary


def test_normalisation():
    # Setup
    desired = 0
    Loader = DataLoader()
    columns = ["Age", "Resting Blood Pressure", "Cholestoral", "Maximum Heart Rate", "ST Depression", "Number of Major Vessels"]

    dataset = Loader._ingest_data()
    dataset = Loader._handle_missing_data(dataset)
    dataset = Loader._handle_categorical(dataset)

    # Exercise
    dataset = Loader._apply_normalisation(dataset)

    actual = 0
    for column in columns:
        actual += len(dataset.loc[dataset[column] > 1, column])

    # Verify
    assert desired == actual

    # Cleanup - none necessary


def test_prepare_dataset():
    # Setup - none necessary
    Loader = DataLoader()
    dataset = Loader.dataset

    # Exercise
    summary = dataset.describe(include="all")

    # Verify
    assert all([a == 1 for a in summary.loc["max"]])
    assert all([a == 303 for a in summary.loc["count"]])


def test_balance_data():
    # Setup
    Loader = DataLoader()
    dataset = Loader.dataset

    # Exercise
    dataset = Loader.balance_data(dataset)
    heart_disease = len(dataset[dataset["Heart Disease"] == 1])
    no_heart_disease = len(dataset[dataset["Heart Disease"] == 0])

    # Verify
    assert heart_disease == no_heart_disease
