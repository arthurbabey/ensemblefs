from collections import defaultdict
from itertools import combinations
from typing import List, Optional, Set

import numpy as np

from ..core.feature import Feature
from .base_merger import MergingStrategy


class UnionOfIntersectionsMerger(MergingStrategy):
    """Set-based merging strategy computing the union of pairwise subset intersections."""

    name = "UnionOfIntersections"

    def __init__(self) -> None:
        super().__init__("set-based")

    def merge(
        self,
        subsets: List[List[Feature]],
        num_features_to_select: Optional[int] = None,
        fill: bool = False,
        **kwargs,
    ) -> Set[str]:
        """
        Merges input subsets by computing the union of pairwise subset intersections.

        Args:
            subsets: List of subsets, where each subset is a list of Feature objects.
            num_features_to_select: Number of selected features (only used if `fill=True`).
            fill: If True, ensures exactly `num_features_to_select` features. Otherwise, returns raw core set.
            **kwargs: Additional strategy-specific arguments.

        Returns:
            A set of selected feature names.

        Raises:
            ValueError: If input subsets are invalid or `num_features_to_select` is missing when `fill=True`.
        """
        self._validate_input(subsets)

        if fill and num_features_to_select is None:
            raise ValueError(
                "`num_features_to_select` must be provided when `fill=True`."
            )

        if len(subsets) == 1:
            feature_names = {f.name for f in subsets[0]}
            return (
                set(
                    sorted(feature_names, key=lambda f: f.score, reverse=True)[
                        :num_features_to_select
                    ]
                )
                if fill
                else feature_names
            )

        # Extract feature names and scores
        feature_names = [[f.name for f in subset] for subset in subsets]
        feature_scores = np.array(
            [[f.score for f in subset] for subset in subsets], dtype=np.float32
        ).T

        # Normalize scores within each subset (vectorized min-max scaling)
        min_vals, max_vals = feature_scores.min(
            axis=1, keepdims=True
        ), feature_scores.max(axis=1, keepdims=True)
        score_range = np.where(
            max_vals - min_vals == 0, 1, max_vals - min_vals
        )  # Prevent division by zero
        feature_scores = (feature_scores - min_vals) / score_range

        # Compute core as the union of pairwise intersections
        core = set().union(
            *[
                set(feature_names[i]) & set(feature_names[j])
                for i, j in combinations(range(len(feature_names)), 2)
            ]
        )

        if not fill:
            return core  # Return raw core without enforcing `num_features_to_select`

        # Compute global feature scores (sum of normalized values)
        feature_score_map = defaultdict(float)
        for subset, scores in zip(feature_names, feature_scores.T):
            for name, score in zip(subset, scores):
                feature_score_map[name] += score

        # Prune or fill to get exactly `num_features_to_select`
        core_list = sorted(core, key=lambda x: feature_score_map[x], reverse=True)
        core_size = len(core_list)

        if core_size >= num_features_to_select:
            return set(core_list[:num_features_to_select])

        # Fill with highest-ranked extra features
        extras = sorted(
            feature_score_map.keys(), key=lambda x: feature_score_map[x], reverse=True
        )
        extras = [f for f in extras if f not in core][
            : num_features_to_select - core_size
        ]

        return set(core_list + extras)
