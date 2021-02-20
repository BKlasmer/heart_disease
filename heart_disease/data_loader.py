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
        self.dataset = self.prepare_dataset()

    
    def prepare_dataset(self) -> pd.DataFrame:
        dataset = self._ingest_data()
        dataset = self._handle_missing_data(dataset)
        dataset = self._handle_categorical(dataset)
        return dataset

    def _ingest_data(self) -> pd.DataFrame:
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

    def _handle_missing_data(self, dataset: pd.DataFrame) -> pd.DataFrame:

        self._logger.info(f"{len(dataset.loc[dataset['Thal'] == '?', 'Thal'])} missing values in Thal, replaced with 3.0 (= normal).")
        dataset.loc[dataset["Thal"] == "?", "Thal"] = '3.0'

        self._logger.info(f"{len(dataset.loc[dataset['Number of Major Vessels'] == '?', 'Number of Major Vessels'])} missing values in Number of Major Vessels, replaced with 0.0 (= mode).")
        dataset.loc[dataset["Number of Major Vessels"] == "?", "Number of Major Vessels"] = '0.0'

        # Change both these column types to floats
        dataset = dataset.astype({"Thal": "float64", "Number of Major Vessels": "float64"})

        return dataset

    def _handle_categorical(self, dataset: pd.DataFrame) -> pd.DataFrame:
        
        one_hot_dict = {
            "Chest Pain Type": ["Chest Pain Typical", "Chest Pain Atypical", "Chest Pain Non-anginal", "Chest Pain Asymptomatic"],
            "Resting ECG": ["Resting ECG Normal", "Resting ECG Abnormal", "Resting ECG Hypertrophy"],
            "Slope of Peak Exercise": ["Peak Exercise Slope Up", "Peak Exercise Slope Flat", "Peak Exercise Slope Down"],
            "Thal": ["Thal Normal", "Thal Fixed Defect", "Thal Reversable Defect"],
        }

        for column, new_columns in one_hot_dict.items():
            temp = pd.get_dummies(dataset[column])
            temp.columns = new_columns
            dataset = dataset.join(temp)
            dataset = dataset.drop(column, axis=1)

        return dataset