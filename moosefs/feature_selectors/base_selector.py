from typing import Any

import numpy as np


class FeatureSelector:
    """Base class for feature selection.

    Subclasses must implement ``compute_scores`` returning a score per feature.
    """

    def __init__(self, task: str, num_features_to_select: int) -> None:
        """Initialize the selector.

        Args:
            task: Either "classification" or "regression".
            num_features_to_select: Number of top features to select.
        """
        self.task = task
        self.num_features_to_select = num_features_to_select

    def select_features(self, X: Any, y: Any) -> tuple:
        """Select top features using the computed scores.

        Args:
            X: Training samples, shape (n_samples, n_features).
            y: Targets, shape (n_samples,) or (n_samples, n_outputs).

        Returns:
            Tuple (scores, indices) where indices are the top-k positions.
        """
        scores = self.compute_scores(X, y)
        indices = np.argsort(scores)[::-1][: self.num_features_to_select]
        return scores, indices

    def compute_scores(self, X: Any, y: Any) -> np.ndarray:
        """Compute per-feature scores (override in subclasses)."""
        raise NotImplementedError("Subclasses must implement compute_scores")
