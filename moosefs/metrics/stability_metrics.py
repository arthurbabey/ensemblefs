from itertools import combinations
from typing import List, Set

from moosefs.core.novovicova import StabilityNovovicova


def compute_stability_metrics(features_list: List[List[str]]) -> float:
    """Compute stability SH(S) across selections.

    Args:
        features_list: Selected feature names per selector.

    Returns:
        Stability in [0, 1].
    """
    return StabilityNovovicova(features_list).compute_stability()


def _jaccard(a: Set[str], b: Set[str]) -> float:
    """Return Jaccard similarity, handling empty sets as 1.0 if both empty."""
    return len(a & b) / len(a | b) if a | b else 1.0


def diversity_agreement(
    selectors: List[List[str]],
    merged: List[str],
    alpha: float = 0.5,
) -> float:
    """Blend diversity and agreement into a single score.

    Args:
        selectors: List of selected feature lists (one per selector).
        merged: Merged/core feature names for the group.
        alpha: Weight on agreement (0 → pure diversity, 1 → pure agreement).

    Returns:
        Score in [0, 1] (higher is better).
    """
    k = len(selectors)
    if k < 2:
        return 0.0  # cannot measure diversity with a single selector

    sets = [set(s) for s in selectors]
    core = set(merged)

    # 1) diversity  (average Jaccard *dissimilarity* across selector pairs)
    pair_dis = [
        1.0 - _jaccard(sets[i], sets[j])
        for i, j in combinations(range(k), 2)
    ]
    diversity = sum(pair_dis) / len(pair_dis)

    # 2) agreement (mean similarity of each selector to the core)
    agree = sum(_jaccard(s, core) for s in sets) / k

    # 3) linear blend
    return (1.0 - alpha) * diversity + alpha * agree
