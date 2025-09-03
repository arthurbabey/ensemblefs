from typing import List
import numpy as np

from ..core.feature import Feature
from .base_merger import MergingStrategy


class ArithmeticMeanMerger(MergingStrategy):
    """Rank-based merging using the arithmetic mean of scores."""

    name = "ArithmeticMean"

    def __init__(self, **kwargs) -> None:
        # Keep taxonomy consistent with existing mergers
        super().__init__("rank-based")
        self.kwargs = kwargs

    def merge(
        self,
        subsets: List[List[Feature]],
        num_features_to_select: int,
        **kwargs,
    ) -> List[str]:
        """Return the top‑k feature names after arithmetic-mean aggregation.

        Args:
            subsets: Feature lists (one list per selector).
            num_features_to_select: Number of names to return.

        Returns:
            Feature names sorted by mean score.
        """
        self._validate_input(subsets)

        # Shortcut if only one selector supplied
        if len(subsets) == 1:
            return [f.name for f in subsets[0]][:num_features_to_select]

        feature_names = [f.name for f in subsets[0]]
        # shape: (n_features, n_selectors)
        scores = np.array([[f.score for f in s] for s in subsets]).T

        # Arithmetic mean across selectors
        scores_merged = scores.mean(axis=1)

        # Lower score ⇒ higher rank (same convention as Borda)
        sorted_names = [
            feature_names[i] for i in np.argsort(-scores_merged, kind="stable")
        ]
        return sorted_names[:num_features_to_select]
