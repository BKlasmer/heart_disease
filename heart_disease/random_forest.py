#!/usr/bin/env python3
# -*- coding: latin-1 -*-

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from heart_disease.utils import Logging

""" Class to train and predict on the heart disease dataset
"""


class RandomForest(object):
    def __init__(self, logger_level: str = "INFO") -> None:
        self._logger = Logging().create_logger(logger_name="Random Forest", logger_level=logger_level)
        self._logger.info("Initialise the Random Forest")

    def train_random_forest_classifier(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        test_features: np.ndarray,
        test_labels: np.ndarray,
        n_estimators: int=1000,
        max_depth: int=None,
        max_features: int=None,
        verbose=False,
        random_state=42,
    ) -> BaseEstimator:
        
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=None,
            n_jobs=-1,
            random_state=random_state,
            verbose=verbose,
        )

        # Train the model on training data
        rf.fit(train_features, train_labels)
        score = rf.score(test_features, test_labels)
        if verbose:
            self._logger.info(f"Testing Set Score: {score:.4f}")
        return rf, score
