from itertools import combinations
from typing import List

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


from itertools import combinations


def jaccard_similarity(a, b):
    return len(a & b) / len(a | b) if a | b else 1.0


def jaccard_dissimilarity(a, b):
    return 1 - jaccard_similarity(a, b)


def compute_diversity_agreement(selected_sets, merged_set):
    # Convert to sets
    selected_sets = [set(s) for s in selected_sets]
    merged_set = set(merged_set)

    k = len(selected_sets)
    if k < 2:
        return 0.0  # Cannot compute diversity with less than 2 sets

    # Diversity: average pairwise Jaccard dissimilarity
    diversity_scores = [
        jaccard_dissimilarity(selected_sets[i], selected_sets[j])
        for i, j in combinations(range(k), 2)
    ]
    diversity = sum(diversity_scores) / len(diversity_scores)

    # Agreement: average similarity with merged set
    agreement_scores = [jaccard_similarity(s, merged_set) for s in selected_sets]
    agreement = sum(agreement_scores) / k

    return diversity * agreement
