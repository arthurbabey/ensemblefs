import numpy as np
import pandas as pd
from mrmr import mrmr_classif, mrmr_regression

from .base_selector import FeatureSelector


class MRMRSelector(FeatureSelector):
    """
    A feature selector using Minimum Redundancy Maximum Relevance (MRMR) for feature scoring.
    This class extends the FeatureSelector base class and computes a score for each feature.

    Attributes:
        task (str): Specifies the machine learning task, either 'classification' or 'regression'.
        kwargs (dict): Additional keyword arguments to pass to mRMR functions.
    """

    name = "MRMR"

    def __init__(self, task, num_features_to_select=None, **kwargs):
        """
        Initializes the MRMRSelector with the specified task and additional parameters.

        Args:
            task (str): The machine learning task ("classification" or "regression").
            num_features_to_select (int, optional): The number of features to select.
            **kwargs: Arbitrary keyword arguments that are passed to mrmr_classif or mrmr_regression.
        """
        super().__init__(task, num_features_to_select)
        self.kwargs = kwargs

    def compute_scores(self, X, y):
        """
        Computes feature scores using the MRMR algorithm.

        Args:
            X (array-like or DataFrame): Training input samples.
            y (array-like or Series): Target values.

        Returns:
            list: A list of scores, one for each feature, in the same order as the columns of X.
        """
        # Ensure X is a DataFrame
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        elif not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame or a NumPy array.")

        # Ensure y is a Series
        if isinstance(y, np.ndarray):
            y = pd.Series(y)

        # Select MRMR function based on task
        if self.task == "classification":
            _, relevance, redundancy = mrmr_classif(
                X, y, K=X.shape[1], return_scores=True, **self.kwargs
            )
        elif self.task == "regression":
            _, relevance, redundancy = mrmr_regression(
                X, y, K=X.shape[1], return_scores=True, **self.kwargs
            )
        else:
            raise ValueError("Task must be 'classification' or 'regression'")

        # Compute MRMR scores (Relevance / Redundancy)
        mrmr_scores = relevance / redundancy.mean(axis=1).replace(
            0, np.nan
        )  # Avoid division by zero
        mrmr_scores = mrmr_scores.fillna(0)  # Replace NaNs with 0 if needed

        # Return scores in the same order as X.columns
        return [mrmr_scores.get(feature, 0) for feature in X.columns]
