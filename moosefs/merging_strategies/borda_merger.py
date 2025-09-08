import numpy as np
from ranky import borda

from ..core.feature import Feature
from .base_merger import MergingStrategy


class BordaMerger(MergingStrategy):
    """Rank-based merging using the Borda count method."""

    name = "Borda"

    def __init__(self, **kwargs) -> None:
        """Initialize a rank-based merger.

        Args:
            **kwargs: Forwarded to the Borda routine (if applicable).
        """
        super().__init__("rank-based")
        self.kwargs = kwargs

    def merge(
        self, subsets: list, num_features_to_select: int, **kwargs
    ) -> list:
        """Merge by Borda and return top-k names.

        Args:
            subsets: Feature lists (one list per selector).
            num_features_to_select: Number of names to return.

        Returns:
            Feature names sorted by merged Borda scores.
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
