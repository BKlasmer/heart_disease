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
    dataset = Loader.ingest_data()

    # Verify
    assert len(dataset) == desired_rows
    assert len(dataset.columns) == desired_columns
    # Cleanup - none necessary
