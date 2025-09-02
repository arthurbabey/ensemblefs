from itertools import combinations
from typing import List, Set
from ensemblefs.core.novovicova import StabilityNovovicova


def compute_stability_metrics(features_list: List[List[str]]) -> float:
    """
    Computes stability metrics using StabilityNovovicova.

    Args:
        features_list: A list of lists, where each sublist contains selected feature names.

    Returns:
        The computed stability measure SH(S).
    """
    return StabilityNovovicova(features_list).compute_stability()


def _jaccard(a: Set[str], b: Set[str]) -> float:
    """Jaccard similarity; handles empty sets."""
    return len(a & b) / len(a | b) if a | b else 1.0


def diversity_agreement(
    selectors: List[List[str]],
    merged: List[str],
    alpha: float = 0.5,
) -> float:
    """
    Diversity-Agreement score in **[0, 1]** (larger = better).

    Parameters
    ----------
    selectors : list[list[str]]
        Each inner list is the feature subset selected by one method.
    merged : list[str]
        The merged/core features for the group (union-of-intersections, Borda, …).
    alpha : float, default 0.5
        Weight of the *agreement* part.
        • alpha = 0 → pure diversity  
        • alpha = 1 → pure agreement
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
