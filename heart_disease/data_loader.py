#!/usr/bin/env python3
# -*- coding: latin-1 -*-

import pandas as pd
from heart_disease.utils import Logging

""" Class to ingest UCI Heart Disease Data Set - https://archive.ics.uci.edu/ml/datasets/Heart+Disease
"""

class DataLoader(object):
    def __init__(self, logger_level : str = "INFO") -> None:
        self._logger = Logging().create_logger(logger_name="Data Loader", logger_level=logger_level)    
        self._logger.info("Initialise Data Loader")
        self.dataset = self.ingest_data()


    def ingest_data(self) -> pd.DataFrame:
        """Ingests the processed Cleveland dataset from https://archive.ics.uci.edu/ml/datasets/Heart+Disease

        Returns:
            pd.DataFrame: The processed Cleveland dataset with column names
        """
        column_names = [
            "Age",
            "Sex",
            "Chest Pain Type",
            "Resting Blood Pressure",
            "Cholestoral",
            "Fasting Blood Sugar",
            "Resting ECG",
            "Maximum Heart Rate",
            "Exercise Induced Angina",
            "ST Depression",
            "Slope of Peak Exercise",
            "Number of Major Vessels",
            "Thal",
            "Heart Disease",
        ]
        dataset = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data", header = None, names=column_names)
        self._logger.info(f"Dataset loaded: {len(dataset.columns)} columns, {len(dataset)} rows")
        return dataset