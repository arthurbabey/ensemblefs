from typing import Dict, List, Union

import numpy as np
import pandas as pd
from mrmr import mrmr_classif, mrmr_regression

from .base_selector import FeatureSelector


class MRMRSelector(FeatureSelector):
    """Feature selector using Minimum Redundancy Maximum Relevance (MRMR)."""

    name = "MRMR"

    def __init__(self, task: str, num_features_to_select: int, **kwargs: Dict) -> None:
        """
        Args:
            task: ML task ('classification' or 'regression').
            num_features_to_select: Number of features to select.
            **kwargs: Additional arguments for mRMR functions.
        """
        super().__init__(task, num_features_to_select)
        self.kwargs = kwargs

    def compute_scores(
        self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]
    ) -> np.ndarray:
        """
        Computes feature scores using the MRMR algorithm.

        Args:
            X: Training samples.
            y: Target values.

        Returns:
            MRMR scores for each feature.
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        elif not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame or a NumPy array.")

        if isinstance(y, np.ndarray):
            y = pd.Series(y)

        score_func = {
            "classification": mrmr_classif,
            "regression": mrmr_regression,
        }.get(self.task)
        if score_func is None:
            raise ValueError("Task must be 'classification' or 'regression'.")

        _, relevance, redundancy = score_func(
            X, y, K=X.shape[1], return_scores=True, **self.kwargs
        )

        # Compute MRMR scores (Relevance / Mean Redundancy), handling division by zero
        mrmr_scores = relevance / redundancy.mean(axis=1).replace(0, np.nan)
        mrmr_scores = mrmr_scores.fillna(0)
        scores = np.array([mrmr_scores.get(feature, 0) for feature in X.columns])
        return scores
