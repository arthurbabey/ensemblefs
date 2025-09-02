from typing import Dict, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from .base_selector import FeatureSelector


class RandomForestSelector(FeatureSelector):
    """Feature selector using RandomForest feature importance."""

    name = "RandomForest"

    def __init__(self, task: str, num_features_to_select: int, **kwargs: Dict) -> None:
        """
        Args:
            task: ML task ('classification' or 'regression').
            num_features_to_select: Number of features to select.
            **kwargs: Additional arguments for RandomForest model.
        """
        super().__init__(task, num_features_to_select)
        self.kwargs = kwargs

    def compute_scores(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series, pd.DataFrame],
    ) -> np.ndarray:
        """
        Computes feature importances using a RandomForest model.

        Args:
            X: Training samples.
            y: Target values.

        Returns:
            Feature importances from the trained RandomForest model.

        Raises:
            ValueError: If task is not 'classification' or 'regression'.
        """
        model_cls = {
            "classification": RandomForestClassifier,
            "regression": RandomForestRegressor,
        }.get(self.task)
        if model_cls is None:
            raise ValueError("Task must be 'classification' or 'regression'.")

        model = model_cls()
        model.fit(X, y)
        scores = model.feature_importances_
        return scores
