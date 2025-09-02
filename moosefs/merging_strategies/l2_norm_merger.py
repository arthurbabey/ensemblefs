# arithmetic_mean_merger.py  (or add below BordaMerger in the same module)

from typing import List
import numpy as np

from ..core.feature import Feature
from .base_merger import MergingStrategy


class L2NormMerger(MergingStrategy):
    """Rank-based merging strategy using the L2-norm (RMS) of scores."""

    name = "L2Norm"

    def __init__(self, **kwargs) -> None:
        super().__init__("rank-based")
        self.kwargs = kwargs

    def merge(
        self,
        subsets: List[List[Feature]],
        num_features_to_select: int,
        **kwargs,
    ) -> List[str]:
        """Return the top‐k feature names after L2-norm aggregation."""
        self._validate_input(subsets)

        if len(subsets) == 1:
            return [f.name for f in subsets[0]][:num_features_to_select]

        feature_names = [f.name for f in subsets[0]]
        scores = np.array([[f.score for f in s] for s in subsets]).T

        # Euclidean norm (root-mean-square) across selectors
        scores_merged = np.linalg.norm(scores, ord=2, axis=1)

        sorted_names = [
            feature_names[i] for i in np.argsort(-scores_merged, kind="stable")
        ]
        return sorted_names[:num_features_to_select]
