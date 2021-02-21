#!/usr/bin/env python3
# -*- coding: latin-1 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.metrics import roc_curve, auc, fbeta_score
from typing import Tuple
from heart_disease.utils import Logging
from heart_disease import DataLoader

""" Class to train and predict on the heart disease dataset
"""


class RandomForest(object):
    def __init__(self, logger_level: str = "INFO") -> None:
        self._logger = Logging().create_logger(logger_name="Random Forest", logger_level=logger_level)
        self._logger.info("Initialise the Random Forest Class")

    def train(
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
        self._logger.info(f"Testing Set Accuracy: {score:.3f}")

        return rf, score

    def perform_k_fold_cv(self, parameters: dict, dataset: pd.DataFrame, folds: int=10) -> list:

        param_score = []
        for param in ParameterGrid(parameters):

            kfold_generator = KFold(n_splits=folds)
            fold_score = []
            for train_index, test_index in kfold_generator.split(dataset):
                # Create the splits
                train_set = dataset.iloc[train_index, :]
                test_set = dataset.iloc[test_index, :]

                # Balance the dataset
                train_set = DataLoader.balance_data(train_set)
                train_labels, train_features, _ = DataLoader.features_and_labels_to_numpy(train_set)
                test_labels, test_features, _ = DataLoader.features_and_labels_to_numpy(test_set)

                # Train the model
                _, score = self.train(train_features, train_labels, test_features, test_labels, param["n_estimators"], param["max_depth"], param["max_features"])
                fold_score.append(score)

            self._logger.info(f"{folds}-fold Result. n_estimators: {param['n_estimators']}, max_depth: {param['max_depth']}, max_features: {param['max_features']}, accuracy: {np.mean(fold_score):.2f} +/- {np.std(fold_score):.2f}")
            param_score.append((np.mean(fold_score), np.std(fold_score), param["n_estimators"], param["max_depth"], param["max_features"]))

        return param_score

    def evaluate_model(self, model: BaseEstimator, test_features: np.ndarray, test_labels: np.ndarray, betas: list) -> float:

        predicted_probabilities = model.predict_proba(test_features)[:,1]
        fpr, tpr, thresholds = roc_curve(test_labels, predicted_probabilities)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=[15,6])
        plt.subplot(1,2,1)
        plt.plot(fpr, tpr, color='darkorange', label=f"ROC Curve (area = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([-0.02, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")

        plt.subplot(1,2,2)
        x_range = np.linspace(0, 1, 100)
        for beta in betas:
            f_beta, x_range = self.evaluate_fbeta(test_labels, predicted_probabilities, beta, x_range)
            plt.plot(x_range, f_beta, label=f"Beta = {beta}")

        plt.ylim([0.0, 1.05])
        plt.xlabel('Thresholds')
        plt.ylabel('F-Beta Score')
        plt.title('F-Beta')
        plt.legend(loc="lower right")

        return roc_auc

    def plot_feature_importance(self, model: BaseEstimator, feature_list: list) -> None:
        feature_importance = model.feature_importances_
        ind = np.arange(len(feature_importance))
        plt.figure(figsize=[12,8])
        plt.bar(ind, feature_importance)
        plt.xticks(ind, feature_list, rotation='vertical')
        plt.ylabel('Feature Weight')
        plt.title('Feature Importance')

    @staticmethod
    def evaluate_fbeta(test_labels: np.ndarray, predicted_probabilities: np.ndarray, beta: float, x_range: np.ndarray) -> Tuple[list, np.ndarray]:

        f_beta = []
        for threshold in x_range:
            binary_predictions = [1 if x >= threshold else 0 for x in predicted_probabilities]
            f_beta.append(fbeta_score(test_labels, binary_predictions, beta=beta))

        return f_beta, x_range


        
