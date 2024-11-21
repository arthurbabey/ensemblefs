import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import pytest

from ensemblefs.core.novovicova import *
from ensemblefs.core.pareto import ParetoAnalysis


@pytest.fixture
def sample_data():
    # Sample data for testing
    data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    group_names = ["Group 1", "Group 2", "Group 3"]
    return data, group_names


@pytest.fixture
def same_data():
    # Sample data for testing
    data = [[5, 5, 5], [5, 5, 5], [5, 5, 5]]
    group_names = ["Group 1", "Group 2", "Group 3"]
    return data, group_names


@pytest.fixture
def big_data():
    # Sample data for testing with 10 groups
    data = [
        [0, 0, 0],  # 0, 9, -9
        [2, 4, 6],  # 2, 6, -4
        [2, 4, 6],  # 2, 6, -4
        [2, 4, 7],  # 4, 4, 0
        [5, 7, 9],  # 7, 2, 5
        [2, 3, 4],  # 1, 8, -7
        [6, 8, 10],  # 8, 1, 7
        [3, 5, 7],  # 5, 3, 2
        [4, 5, 6],  # 4, 3, 1
        [13, 13, 13],  # 9, 0, 9
    ]
    group_names = [
        "Group 1",
        "Group 2",
        "Group 3",
        "Group 4",
        "Group 5",
        "Group 6",
        "Group 7",
        "Group 8",
        "Group 9",
        "Group 10",
    ]
    return data, group_names


def test_compute_dominance(sample_data):
    data, group_names = sample_data
    pareto = ParetoAnalysis(data, group_names)
    pareto.compute_dominance()

    assert pareto.results == [
        ["Group 1", 0, 2, -2],
        ["Group 2", 1, 1, 0],
        ["Group 3", 2, 0, 2],
    ]


def test_get_results(sample_data):
    data, group_names = sample_data
    pareto = ParetoAnalysis(data, group_names)
    results = pareto.get_results()
    expected_results = [
        ["Group 3", 2, 0, 2],
        ["Group 2", 1, 1, 0],
        ["Group 1", 0, 2, -2],
    ]

    assert results == expected_results


def test_compute_dominance_same(same_data):
    data, group_names = same_data
    pareto = ParetoAnalysis(data, group_names)
    pareto.compute_dominance()

    assert pareto.results == [
        ["Group 1", 0, 0, 0],
        ["Group 2", 0, 0, 0],
        ["Group 3", 0, 0, 0],
    ]


# potential issue when sorting, how to choose best group when scalar is equal
def test_get_results_same(same_data):
    data, group_names = same_data
    pareto = ParetoAnalysis(data, group_names)
    results = pareto.get_results()
    expected_results = [
        ["Group 1", 0, 0, 0],
        ["Group 2", 0, 0, 0],
        ["Group 3", 0, 0, 0],
    ]

    assert results == expected_results


def test_big_data(big_data):
    data, group_names = big_data
    pareto = ParetoAnalysis(data, group_names)
    pareto.compute_dominance()
    results = pareto.get_results()
    sorted_expected_results = [
        ["Group 10", 9, 0, 9],
        ["Group 7", 8, 1, 7],
        ["Group 5", 7, 2, 5],
        ["Group 8", 5, 3, 2],
        ["Group 9", 4, 3, 1],
        ["Group 4", 4, 4, 0],
        ["Group 2", 2, 6, -4],
        ["Group 3", 2, 6, -4],
        ["Group 6", 1, 8, -7],
        ["Group 1", 0, 9, -9],
    ]

    expected_results = [
        ["Group 1", 0, 9, -9],
        ["Group 2", 2, 6, -4],
        ["Group 3", 2, 6, -4],
        ["Group 4", 4, 4, 0],
        ["Group 5", 7, 2, 5],
        ["Group 6", 1, 8, -7],
        ["Group 7", 8, 1, 7],
        ["Group 8", 5, 3, 2],
        ["Group 9", 4, 3, 1],
        ["Group 10", 9, 0, 9],
    ]
    assert pareto.results == expected_results
    assert results == sorted_expected_results
