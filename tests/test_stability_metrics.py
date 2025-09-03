import pytest

from moosefs.metrics.stability_metrics import (
    compute_stability_metrics,
    diversity_agreement,
)
from moosefs.core.novovicova import StabilityNovovicova


def test_compute_stability_metrics_matches_core():
    selectors = [["a", "b"], ["b", "c"], ["a", "c"]]
    # wrapper should match direct computation
    expected = StabilityNovovicova([set(s) for s in selectors]).compute_stability()
    got = compute_stability_metrics(selectors)
    assert got == expected


def test_diversity_agreement_limits_and_edges():
    # fewer than two selectors → 0.0
    assert diversity_agreement([["a", "b"]], ["a"]) == 0.0

    # identical selectors and core → high agreement, low diversity
    sels = [["a", "b"], ["a", "b"]]
    score = diversity_agreement(sels, ["a", "b"], alpha=0.5)
    assert 0.0 <= score <= 1.0

    # dissimilar selectors and empty core
    sels = [["a"], ["b"]]
    score = diversity_agreement(sels, [], alpha=0.0)  # pure diversity
    assert 0.0 <= score <= 1.0

