from collections import defaultdict
from itertools import combinations

import numpy as np

from .base_merger import MergingStrategy


class UnionOfIntersectionsMerger(MergingStrategy):
    """
    A set-based merging strategy that computes the union of intersections of pairs of subsets.
    This class extends the MergingStrategy base class.
    """

    name = "UnionOfIntersections"

    def __init__(self):
        super().__init__("set-based")

    def merge(self, subsets, num_features=None, **kwargs):
        """
        Merges the input subsets by computing the union of intersections of pairs of subsets,
        then ensures the final set has exactly `num_features` features (if specified) by ranking
        all features (via summed scores across subsets).

        Args:
            subsets (list of lists): A list of subsets (each subset is a list of Feature objects).
            num_features (int, optional): Desired number of features in the final set.
            **kwargs: Additional keyword arguments specific to the merging strategy.

        Returns:
            list: The final list of selected feature names, of size `num_features` if provided.

        Raises:
            ValueError: If subset does not contain Feature objects.
        """

        self._validate_input(subsets)

        if len(subsets) == 1:
            if num_features is None:
                return [f.name for f in subsets[0]]
            return sorted(
                [f.name for f in subsets[0]], key=lambda f: f.score, reverse=True
            )[:num_features]

        # Step 1: Extract feature names and scores efficiently
        feature_names = [[f.name for f in subset] for subset in subsets]
        feature_scores = np.array(
            [[f.score for f in subset] for subset in subsets], dtype=np.float32
        ).T

        # Step 2: Normalize scores within each subset (vectorized min-max scaling)
        min_vals = feature_scores.min(axis=1, keepdims=True)
        max_vals = feature_scores.max(axis=1, keepdims=True)
        score_range = max_vals - min_vals
        score_range[score_range == 0] = 1  # Prevent division by zero
        feature_scores = (feature_scores - min_vals) / score_range

        # Step 3: Compute "core" as the union of pairwise intersections
        core = set().union(
            *[
                set(feature_names[i]) & set(feature_names[j])
                for i, j in combinations(range(len(feature_names)), 2)
            ]
        )
        if num_features is None:
            return list(core)

        # Step 4: Compute global feature scores (sum of normalized values)
        feature_score_map = defaultdict(float)
        for subset, scores in zip(feature_names, feature_scores.T):
            for name, score in zip(subset, scores):
                feature_score_map[name] += score

        # Step 5: Prune or fill to get exactly num_features
        core_list = sorted(core, key=lambda x: feature_score_map[x], reverse=True)
        core_size = len(core_list)

        if core_size == num_features:
            return core_list
        elif core_size > num_features:
            return core_list[:num_features]
        else:
            needed = num_features - core_size
            all_sorted = sorted(
                feature_score_map.keys(),
                key=lambda x: feature_score_map[x],
                reverse=True,
            )
            extras = [f for f in all_sorted if f not in core][:needed]
            return core_list + extras
