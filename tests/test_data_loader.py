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

    # Exercise
    dataset = Loader.dataset

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
    columns = ["Chest Pain Typical", "Chest Pain Atypical", "Chest Pain Non-anginal", "Chest Pain Asymptomatic", "Resting ECG Normal", "Resting ECG Abnormal", "Resting ECG Hypertrophy", "Peak Exercise Slope Up", "Peak Exercise Slope Flat", "Peak Exercise Slope Down", "Thal Normal", "Thal Fixed Defect", "Thal Reversable Defect"]

    # Exercise
    dataset = Loader.dataset
    
    actual = 0
    for column in columns:
        actual += len(dataset.loc[dataset[column] > 1, column])

    # Verify
    assert desired == actual

    # Cleanup - none necessary
    
