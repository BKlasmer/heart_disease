#!/usr/bin/env python3
# -*- coding: latin-1 -*-

import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from heart_disease.utils import Logging


""" Class to ingest UCI Heart Disease Data Set - https://archive.ics.uci.edu/ml/datasets/Heart+Disease
"""


class DataLoader(object):
    def __init__(self, logger_level: str = "INFO") -> None:
        self._logger = Logging().create_logger(logger_name="Data Loader", logger_level=logger_level)
        self._logger.info("Initialise Data Loader")
        self.dataset = self.prepare_dataset()

    def prepare_dataset(self) -> pd.DataFrame:
        """Prepares the heart disease dataset, which performs the following:
        - Ingesting the data
        - Handling missing data
        - One hot encoding the categorical data
        - Apply normalisation
        - Binarise the label

        Returns:
            pd.DataFrame: Heart disease dataset
        """
        dataset = self._ingest_data()
        dataset = self._handle_missing_data(dataset)
        dataset = self._handle_categorical(dataset)
        dataset = self._apply_normalisation(dataset)

        # Change heart disease to binary
        heart_disease = pd.DataFrame(
            [1 if x >= 1 else 0 for x in dataset["Heart Disease"].to_list()], columns=["Heart Disease"]
        )
        dataset = dataset.drop("Heart Disease", axis=1)
        dataset = dataset.join(heart_disease)

        return dataset

    def split_dataset(self, test_size: float = 0.2, balance: bool = True, random_state: int = 42):
        train, test = train_test_split(self.dataset, test_size=test_size, random_state=random_state)
        if balance:
            train = self._balance_data(train, random_state)

        train_labels, train_features, _ = self._features_and_labels_to_numpy(train)
        test_labels, test_features, _ = self._features_and_labels_to_numpy(test)

        self._logger.info(f"Training Features Shape: {train_features.shape}")
        self._logger.info(f"Training Labels Shape: {train_labels.shape}")
        self._logger.info(f"Testing Features Shape: {test_features.shape}")
        self._logger.info(f"Testing Labels Shape: {test_labels.shape}")

        return train_features, train_labels, test_features, test_labels

    def _balance_data(self, train_set: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
        heart_disease = train_set[train_set["Heart Disease"] == 1]
        no_heart_disease = train_set[train_set["Heart Disease"] == 0]

        max_samples = max(len(heart_disease), len(no_heart_disease))

        # Upsample to balance
        if len(heart_disease) < len(no_heart_disease):
            heart_disease = resample(heart_disease, replace=True, n_samples=max_samples, random_state=random_state)
        else:
            no_heart_disease = resample(no_heart_disease, replace=True, n_samples=max_samples, random_state=random_state)

        train_set = pd.concat([heart_disease, no_heart_disease])

        return train_set

    def _features_and_labels_to_numpy(self, dataset):
        labels = np.array(dataset["Heart Disease"])

        dataset = dataset.drop("Heart Disease", axis=1)
        features = np.array(dataset)
        feature_columns = list(dataset.columns)
        return labels, features, feature_columns

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
        dataset = pd.read_csv(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
            header=None,
            names=column_names,
        )
        self._logger.info(f"Dataset loaded: {len(dataset.columns)} columns, {len(dataset)} rows")
        return dataset

    def _handle_missing_data(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Replace all missing data in the dataset

        Args:
            dataset (pd.DataFrame): Heart disease dataset with missing values

        Returns:
            pd.DataFrame: Heart disease dataset without missing values
        """

        self._logger.info(
            f"{len(dataset.loc[dataset['Thal'] == '?', 'Thal'])} missing values in Thal, replaced with 3.0 (= normal)."
        )
        dataset.loc[dataset["Thal"] == "?", "Thal"] = "3.0"

        self._logger.info(
            f"{len(dataset.loc[dataset['Number of Major Vessels'] == '?', 'Number of Major Vessels'])} missing values in Number of Major Vessels, replaced with 0.0 (= mode)."
        )
        dataset.loc[dataset["Number of Major Vessels"] == "?", "Number of Major Vessels"] = "0.0"

        # Change both these column types to floats
        dataset = dataset.astype({"Thal": "float64", "Number of Major Vessels": "float64"})

        return dataset

    def _handle_categorical(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """One hot encodes all the categorical fields in the dataset

        Args:
            dataset (pd.DataFrame): Heart disease dataset with categorical features

        Returns:
            pd.DataFrame: Heart disease dataset with one-hot encoded categorical features
        """

        one_hot_dict = {
            "Chest Pain Type": [
                "Chest Pain Typical",
                "Chest Pain Atypical",
                "Chest Pain Non-anginal",
                "Chest Pain Asymptomatic",
            ],
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

    def _apply_normalisation(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Normalises the dataset to values between 0 and 1

        Args:
            dataset (pd.DataFrame): Heart disease dataset with unbounded features

        Returns:
            pd.DataFrame: Heart disease dataset with features bounded between 0 and 1
        """

        variable_columns = [
            "Age",
            "Resting Blood Pressure",
            "Cholestoral",
            "Maximum Heart Rate",
            "ST Depression",
            "Number of Major Vessels",
        ]

        for column in variable_columns:
            column_values = dataset[column].to_numpy()
            dataset[column] = self._minmax(column_values)

        return dataset

    @staticmethod
    def _minmax(column_values: np.ndarray) -> np.ndarray:
        """Applies min max normalisation on a numpy array

        Args:
            column_values (np.ndarray): Unbounded numpy array

        Returns:
            np.ndarray: Min max normalised numpy array
        """
        min_val = np.min(column_values)
        max_val = np.max(column_values)

        return (column_values - min_val) / (max_val - min_val)
