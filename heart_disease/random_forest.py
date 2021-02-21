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
        """Train a random forest

        Args:
            train_features (np.ndarray): Features to train on
            train_labels (np.ndarray): Labels to train on 
            test_features (np.ndarray): Features to test on
            test_labels (np.ndarray): Labels to test on
            n_estimators (int, optional): Number of trees in the forest. Defaults to 1000.
            max_depth (int, optional): Maximum depth of each tree. Defaults to None.
            max_features (int, optional): Maximum number of features to use for each tree. Defaults to None.
            verbose (bool, optional): Defaults to False.
            random_state (int, optional): Random Seed. Defaults to 42.

        Returns:
            BaseEstimator: Trained random forest
        """
        
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
        """Perform K-fold cross validation with a parameter grid search. For each fold, the training set is balanced.

        Args:
            parameters (dict): Parameters to search over
            dataset (pd.DataFrame): Dataset to run cross validation over
            folds (int, optional): Number of folds. Defaults to 10.

        Returns:
            list: Average k-fold cross validation accuracy and standard deviation for each parameter combination
        """

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
        """Plot AUC and calculate AUROC. Plot F-beta scores over a range of thresholds and beta values.

        Args:
            model (BaseEstimator): Trained random forest model to evaluate
            test_features (np.ndarray): Features to test on
            test_labels (np.ndarray): Labels to test on
            betas (list): Beta values to evaluate against

        Returns:
            float: AUROC score
        """
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
        """Plots the feature importance

        Args:
            model (BaseEstimator): Trained random forest model
            feature_list (list): List of feature names
        """
        feature_importance = model.feature_importances_
        ind = np.arange(len(feature_importance))
        plt.figure(figsize=[12,8])
        plt.bar(ind, feature_importance)
        plt.xticks(ind, feature_list, rotation='vertical')
        plt.ylabel('Feature Weight')
        plt.title('Feature Importance')

    @staticmethod
    def evaluate_fbeta(test_labels: np.ndarray, predicted_probabilities: np.ndarray, beta: float, x_range: np.ndarray) -> Tuple[list, np.ndarray]:
        """Evaluate F-beta score

        Args:
            test_labels (np.ndarray): Labels to test on
            predicted_probabilities (np.ndarray): Predicted probabilities of test set
            beta (float): Beta value to evaluate
            x_range (np.ndarray): Range of thresholds to test

        Returns:
            Tuple[list, np.ndarray]: List of F-beta scores, and array of thresholds tested
        """
        f_beta = []
        for threshold in x_range:
            binary_predictions = [1 if x >= threshold else 0 for x in predicted_probabilities]
            f_beta.append(fbeta_score(test_labels, binary_predictions, beta=beta))

        return f_beta, x_range


        
