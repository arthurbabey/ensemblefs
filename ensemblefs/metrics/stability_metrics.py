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
