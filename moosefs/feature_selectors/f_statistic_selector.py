from typing import Dict, Union

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif, f_regression

from .base_selector import FeatureSelector


class FStatisticSelector(FeatureSelector):
    """Feature selector using F-statistic scores."""

    name = "FStatistic"

    def __init__(self, task: str, num_features_to_select: int, **kwargs: Dict) -> None:
        """
        Args:
            task: ML task ('classification' or 'regression').
            num_features_to_select: Number of features to select.
            **kwargs: Additional arguments for the scoring function.
        """
        super().__init__(task, num_features_to_select)
        self.kwargs = kwargs

    def compute_scores(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series, pd.DataFrame],
    ) -> np.ndarray:
        """
        Computes F-statistic scores.

        Args:
            X: Training samples.
            y: Target values.

        Returns:
            F-statistic scores for each feature.

        Raises:
            ValueError: If task is not 'classification' or 'regression'.
        """
        score_func = {"classification": f_classif, "regression": f_regression}.get(
            self.task
        )
        if score_func is None:
            raise ValueError("Task must be 'classification' or 'regression'.")
        scores, _ = score_func(X, y, **self.kwargs)
        return scores
