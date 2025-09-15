import numpy as np

from .base_merger import MergingStrategy


class L2NormMerger(MergingStrategy):
    """Rank-based merging using the L2-norm (RMS) of scores."""

    name = "L2Norm"

    def __init__(self, **kwargs) -> None:
        super().__init__("rank-based")
        self.kwargs = kwargs

    def merge(
        self,
        subsets: list,
        num_features_to_select: int,
        **kwargs,
    ) -> list:
        """Return the top‑k feature names after L2-norm aggregation.

        Args:
            subsets: Feature lists (one list per selector).
            num_features_to_select: Number of names to return.

        Returns:
            Feature names sorted by aggregated L2 score.
        """
        self._validate_input(subsets)

        if len(subsets) == 1:
            return [f.name for f in subsets[0]][:num_features_to_select]

        feature_names = [f.name for f in subsets[0]]
        scores = np.array([[f.score for f in s] for s in subsets]).T

        # Euclidean norm (root-mean-square) across selectors
        scores_merged = np.linalg.norm(scores, ord=2, axis=1)

        sorted_names = [feature_names[i] for i in np.argsort(-scores_merged, kind="stable")]
        return sorted_names[:num_features_to_select]
