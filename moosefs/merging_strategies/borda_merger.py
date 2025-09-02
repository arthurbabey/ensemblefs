from typing import List

import numpy as np
from ranky import borda

from ..core.feature import Feature
from .base_merger import MergingStrategy


class BordaMerger(MergingStrategy):
    """Rank-based merging strategy using the Borda count method."""

    name = "Borda"

    def __init__(self, **kwargs) -> None:
        """
        Initializes the BordaMerger with 'rank-based' type.

        Args:
            **kwargs: Additional arguments for the Borda count method.
        """
        super().__init__("rank-based")
        self.kwargs = kwargs

    def merge(
        self, subsets: List[List[Feature]], num_features_to_select: int, **kwargs
    ) -> List[str]:
        """
        Merges subsets using the Borda count method.

        Args:
            subsets: List of subsets, where each subset contains Feature objects.
            num_features_to_select: Number of top-ranked features to return.

        Returns:
            A list of selected feature names, sorted by merged Borda scores.
        """
        self._validate_input(subsets)

        if len(subsets) == 1:
            return [feature.name for feature in subsets[0]][:num_features_to_select]

        # Extract feature names (from the first subset) and scores
        feature_names = [feature.name for feature in subsets[0]]
        scores = np.array(
            [[feature.score for feature in subset] for subset in subsets]
        ).T

        # Apply Borda count method
        scores_merged = borda(m=scores, **self.kwargs)

        # Sort based on Borda scores (lower score = higher rank)
        sorted_names = [
            feature_names[i] for i in np.argsort(scores_merged, kind="stable")
        ]

        return list(sorted_names[:num_features_to_select])
