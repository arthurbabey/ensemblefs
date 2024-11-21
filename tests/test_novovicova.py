import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from ensemblefs.core.novovicova import *


def test_empty_input():
    with pytest.raises(ValueError):
        StabilityNovovicova([])


def test_non_uniform_data_types():
    with pytest.raises(ValueError):
        StabilityNovovicova([{1, 2, 3}, {4, 5, "six"}])


def test_random_input():
    import numpy as np

    for _ in range(1000):
        feature_selections = [
            list(
                np.random.choice(
                    range(1, 10), size=np.random.randint(1, 5), replace=False
                )
            )
            for _ in range(100)
        ]
        stability_calculator = StabilityNovovicova(feature_selections)
        stability = stability_calculator.compute_stability()
        assert 0 <= stability <= 1, "Stability should be between 0 and 1"


def test_property_zero_stability():
    feature_selections = [{1}, {2}, {3}, {4}]
    stability_calculator = StabilityNovovicova(feature_selections)
    stability = stability_calculator.compute_stability()
    assert (
        stability == 0
    ), "Stability should be 0 when each feature appears exactly once"


def test_property_full_stability():
    feature_selections = [{1, 2, 3, 4} for _ in range(5)]
    stability_calculator = StabilityNovovicova(feature_selections)
    stability = stability_calculator.compute_stability()
    assert (
        stability == 1
    ), "Stability should be 1 when each feature appears in every subset"


def test_identical_feature_selections():
    feature_selections = [{1, 2, 3} for _ in range(5)]
    stability_calculator = StabilityNovovicova(feature_selections)
    stability = stability_calculator.compute_stability()
    assert (
        stability == 1
    ), "Stability should be 1 when all feature selections are identical"


def test_identical_feature_selections_strings():
    feature_selections = [{"apple", "banana", "pear"} for _ in range(10)]
    stability_calculator = StabilityNovovicova(feature_selections)
    stability = stability_calculator.compute_stability()
    assert stability == 1, "Same with strings"


def test_completely_different_feature_selections():
    feature_selections = [{i} for i in range(1, 6)]
    stability_calculator = StabilityNovovicova(feature_selections)
    stability = stability_calculator.compute_stability()
    assert (
        stability == 0
    ), "Stability should be 0 when all feature selections are different"


def test_edge_case_single_subset():
    feature_selections = [{1, 2, 3}]
    stability_calculator = StabilityNovovicova(feature_selections)
    stability = stability_calculator.compute_stability()
    assert stability == 0, "Stability should be 0 when there is only one subset"


def test_edge_case_no_features():
    feature_selections = [[], [], []]
    stability_calculator = StabilityNovovicova(feature_selections)
    stability = stability_calculator.compute_stability()
    assert stability == 0, "Stability should be 0 with empty subset"


def test_repeated_features_in_subsets():
    feature_selections = [[1, 2, 2, 3], [1, 3, 3, 4], [1, 1, 4, 4]]
    stability_calculator = StabilityNovovicova(feature_selections)
    stability = stability_calculator.compute_stability()
    assert (
        0 <= stability <= 1
    ), "Stability should be between 0 and 1 even with repeated features in subsets"
