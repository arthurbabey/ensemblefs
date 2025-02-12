import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from ensemblefs.core.feature import Feature
from ensemblefs.merging_strategies import BordaMerger, UnionOfIntersectionsMerger


@pytest.fixture
def union_of_intersections():
    return UnionOfIntersectionsMerger()


@pytest.fixture
def borda_merger():
    return BordaMerger()


def subset_to_Feature(subset):
    return [[Feature(name=str(i), score=i) for i in sublist] for sublist in subset]


def test_basic_functionality(union_of_intersections):
    subsets = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
    subsets = subset_to_Feature(subsets)
    result = union_of_intersections.merge(subsets)
    assert isinstance(result, set)


def test_empty_input(union_of_intersections):
    subsets = []
    with pytest.raises(ValueError):
        union_of_intersections.merge(subsets)


def test_single_subset(union_of_intersections):
    subsets = [[1, 2, 3, 4]]
    subsets = subset_to_Feature(subsets)
    result = union_of_intersections.merge(subsets)
    assert result == {"1", "2", "3", "4"}


def test_multiple_subsets(union_of_intersections):
    subsets = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
    subsets = subset_to_Feature(subsets)
    result = union_of_intersections.merge(subsets)
    assert result == {"2", "3", "4"}


def test_symmetry_property(union_of_intersections):
    subsets_1 = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
    subsets_2 = [[3, 4, 5], [2, 3, 4], [1, 2, 3]]
    subsets_1 = subset_to_Feature(subsets_1)
    subsets_2 = subset_to_Feature(subsets_2)
    result_1 = union_of_intersections.merge(subsets_1)
    result_2 = union_of_intersections.merge(subsets_2)
    assert result_1 == result_2


def test_empty_output(union_of_intersections):
    subsets = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    subsets = subset_to_Feature(subsets)
    result = union_of_intersections.merge(subsets)
    assert result == set()


def test_merge_fill_full(union_of_intersections):
    subsets = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    subsets = subset_to_Feature(subsets)
    result = union_of_intersections.merge(subsets, num_features_to_select=9, fill=True)
    assert result == set([str(i) for i in range(1, 10)])


def test_merge_fill(union_of_intersections):
    subsets = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
    subsets = subset_to_Feature(subsets)
    result = union_of_intersections.merge(subsets, num_features_to_select=4, fill=True)
    for i in ["2", "3", "4"]:
        assert i in result
    assert len(result) == 4


def test_borda_basic_functionality(borda_merger):
    subsets = [
        [Feature("A", 10), Feature("B", 8), Feature("C", 6)],
        [Feature("A", 9), Feature("B", 7), Feature("C", 5)],
        [Feature("A", 11), Feature("B", 6), Feature("C", 4)],
    ]
    result = borda_merger.merge(subsets, num_features_to_select=3)
    assert result == ["A", "B", "C"]  # Expected ranking based on scores


def test_borda_empty_input(borda_merger):
    subsets = []
    with pytest.raises(ValueError):
        borda_merger.merge(subsets, num_features_to_select=3)


def test_borda_single_score_list(borda_merger):
    subsets = [[Feature("A", 10), Feature("B", 8), Feature("C", 6)]]
    result = borda_merger.merge(subsets, num_features_to_select=3)
    assert result == ["A", "B", "C"]  # Single list should return names in order


def test_borda_multiple_scores(borda_merger):
    subsets = [
        [Feature("A", 10), Feature("B", 7), Feature("C", 5)],
        [Feature("A", 9), Feature("B", 8), Feature("C", 6)],
        [Feature("A", 8), Feature("B", 6), Feature("C", 7)],
    ]
    result = borda_merger.merge(subsets, num_features_to_select=3)
    assert result == ["A", "B", "C"]  # Expected merged ranking


def test_borda_k_features(borda_merger):
    subsets = [
        [Feature("A", 10), Feature("B", 8), Feature("C", 6)],
        [Feature("A", 9), Feature("B", 7), Feature("C", 5)],
        [Feature("A", 11), Feature("B", 6), Feature("C", 4)],
    ]
    result = borda_merger.merge(subsets, num_features_to_select=2)
    assert result == ["A", "B"]  # Top 2 features


def test_borda_symmetry_property(borda_merger):
    subsets = [
        [Feature("A", 10), Feature("B", 7), Feature("C", 6)],
        [Feature("A", 7), Feature("B", 10), Feature("C", 6)],
    ]
    result = borda_merger.merge(subsets, num_features_to_select=3)
    assert result == ["A", "B", "C"]  # Symmetry test: A and B tie for top spots


def test_borda_big(borda_merger):
    features1 = [
        Feature(name="A", score=8),
        Feature(name="B", score=3),
        Feature(name="C", score=6),
        Feature(name="D", score=9),
        Feature(name="E", score=1),
        Feature(name="F", score=5),
        Feature(name="G", score=7),
        Feature(name="H", score=2),
        Feature(name="I", score=4),
        Feature(name="J", score=10),
    ]

    features2 = [
        Feature(name="A", score=7),
        Feature(name="B", score=6),
        Feature(name="C", score=5),
        Feature(name="D", score=4),
        Feature(name="E", score=3),
        Feature(name="F", score=10),
        Feature(name="G", score=9),
        Feature(name="H", score=2),
        Feature(name="I", score=8),
        Feature(name="J", score=1),
    ]

    # rank for features 1 :
    # J = 1, D = 2, A =3, G = 4, C = 5, F = 6, I = 7, B = 8, H = 9, E = 10

    # rank for features 2 :
    # F = 1, G = 2, I = 3, A = 4, B = 5, C = 6, D = 7, E = 8, H = 9, J = 10

    # mean borda Rank
    # A = 3.5, B = 6.5, C = 5.5, D = 4.5, E = 9, F = 3.5, G = 3, H = 9, I = 5, J = 5.5

    # ranking from low to high rank
    # G, A, F, D, I, C, J, B, E, H
    expected_result = ["G", "A", "F", "D", "I", "C", "J", "B", "E", "H"]

    subsets = [features1, features2]

    # Call the merge function
    result = borda_merger.merge(subsets, num_features_to_select=10)

    # Assert the result matches the expected output
    assert result == expected_result
