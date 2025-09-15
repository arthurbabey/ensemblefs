import numpy as np
import pytest

from moosefs.core.novovicova import StabilityNovovicova


def test_empty_input():
    with pytest.raises(ValueError, match="Feature selections cannot be empty."):
        StabilityNovovicova([])


def test_non_uniform_data_types():
    with pytest.raises(ValueError, match="All features must be of the same type across selections."):
        StabilityNovovicova([{1, 2, 3}, {4, 5, "six"}])


def test_random_input():
    for _ in range(1000):
        feature_selections = [
            set(np.random.choice(range(1, 10), size=np.random.randint(1, 5), replace=False)) for _ in range(100)
        ]
        stability = StabilityNovovicova(feature_selections).compute_stability()
        assert 0 <= stability <= 1, "Stability should be between 0 and 1"


def test_property_zero_stability():
    feature_selections = [{1}, {2}, {3}, {4}]
    stability = StabilityNovovicova(feature_selections).compute_stability()
    assert stability == 0, "Stability should be 0 when each feature appears exactly once"


def test_property_full_stability():
    feature_selections = [{1, 2, 3, 4} for _ in range(5)]
    stability = StabilityNovovicova(feature_selections).compute_stability()
    assert stability == 1, "Stability should be 1 when each feature appears in every subset"


def test_identical_feature_selections():
    feature_selections = [{1, 2, 3} for _ in range(5)]
    stability = StabilityNovovicova(feature_selections).compute_stability()
    assert stability == 1, "Stability should be 1 when all feature selections are identical"


def test_identical_feature_selections_strings():
    feature_selections = [{"apple", "banana", "pear"} for _ in range(10)]
    stability = StabilityNovovicova(feature_selections).compute_stability()
    assert stability == 1, "Stability should be 1 when all feature selections are identical (strings)"


def test_completely_different_feature_selections():
    feature_selections = [{i} for i in range(1, 6)]
    stability = StabilityNovovicova(feature_selections).compute_stability()
    assert stability == 0, "Stability should be 0 when all feature selections are different"


def test_edge_case_single_subset():
    feature_selections = [{1, 2, 3}]
    stability = StabilityNovovicova(feature_selections).compute_stability()
    assert stability == 0, "Stability should be 0 when there is only one subset"


def test_edge_case_no_features():
    with pytest.raises(ValueError, match="Feature selections cannot contain empty sets."):
        StabilityNovovicova([set(), set(), set()])


def test_repeated_features_in_subsets():
    feature_selections = [
        {1, 2, 2, 3},  # Set will remove duplicates automatically
        {1, 3, 3, 4},
        {1, 1, 4, 4},
    ]
    stability = StabilityNovovicova(feature_selections).compute_stability()
    assert 0 <= stability <= 1, "Stability should be between 0 and 1 even with repeated features in subsets"
