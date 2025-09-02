from typing import List, Set, Union

import numpy as np


class StabilityNovovicova:
    """
    Computes the stability of feature selection algorithms based on Novovicová et al. (2009).

    References:
        Novovicová, J., Somol, P., & Pudil, P. (2009). "A New Measure of Feature Selection
        Algorithms' Stability." IEEE International Conference on Data Mining Workshops.
    """

    def __init__(self, selected_features: List[Union[Set, List]]):
        """
        Args:
            selected_features: A list of sets or lists, where each represents selected features in a dataset.
        """
        self._validate_inputs(selected_features)
        self.selected_features: List[Set] = [
            set(sel) for sel in selected_features
        ]  # Convert all to sets
        self.N: int = sum(
            len(sel) for sel in self.selected_features
        )  # Total feature occurrences
        self.n: int = len(self.selected_features)  # Number of datasets

    @staticmethod
    def _validate_inputs(selected_features: List[Union[Set, List]]) -> None:
        """Validates the input format, ensuring consistency and non-emptiness."""
        if not selected_features:
            raise ValueError("Feature selections cannot be empty.")
        if not isinstance(selected_features, list):
            raise TypeError("Feature selections must be a list of sets or lists.")
        if not all(isinstance(sel, (set, list)) for sel in selected_features):
            raise TypeError("Each feature selection must be a set or a list.")

        if any(len(sel) == 0 for sel in selected_features):
            raise ValueError("Feature selections cannot contain empty sets or lists.")

        # Ensure feature types are consistent
        element_type = type(next(iter(selected_features[0]), None))
        if any(
            any(type(item) != element_type for item in sel) for sel in selected_features
        ):
            raise ValueError("All features must be of the same type across selections.")

    def compute_stability(self) -> float:
        """
        Computes the stability measure SH(S), ranging from 0 (no stability) to 1 (full stability).

        Returns:
            Stability score.
        """
        if self.N == 0 or self.n == 1:
            return 0.0  # Stability is not meaningful for a single subset or empty selection.

        # Count occurrences of each unique feature
        feature_counts: dict = {}
        for sublist in self.selected_features:
            for feature in sublist:
                feature_counts[feature] = feature_counts.get(feature, 0) + 1

        # Compute stability measure
        SH_S: float = sum(count * np.log2(count) for count in feature_counts.values())
        return SH_S / (self.N * np.log2(self.n))
