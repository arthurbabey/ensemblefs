from typing import Tuple, Union

import numpy as np
import pandas as pd


class FeatureSelector:
    """Base class for feature selection."""

    def __init__(self, task: str, num_features_to_select: int) -> None:
        """
        Args:
            task: ML task ('classification' or 'regression').
            num_features_to_select: Number of top features to select.
        """
        self.task = task
        self.num_features_to_select = num_features_to_select

    def select_features(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series, pd.DataFrame],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Selects top features based on computed scores.

        Args:
            X: Training samples, shape [n_samples, n_features].
            y: Target values, shape [n_samples] or [n_samples, n_outputs].

        Returns:
            A tuple containing feature scores and indices of the selected features.
        """
        scores = self.compute_scores(X, y)
        indices = np.argsort(scores)[::-1][: self.num_features_to_select]
        return scores, indices

    def compute_scores(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series, pd.DataFrame],
    ) -> np.ndarray:
        """
        Computes feature scores (to be implemented by subclasses).

        Args:
            X: Training samples.
            y: Target values.

        Returns:
            An array of feature scores.

        Raises:
            NotImplementedError: If not implemented in a subclass.
        """
        raise NotImplementedError("Subclasses must implement compute_scores")
