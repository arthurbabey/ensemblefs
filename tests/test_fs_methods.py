import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from sklearn.datasets import make_classification, make_regression

from ensemblefs.feature_selectors import *


@pytest.fixture
def fake_data_classification():
    informative_features = [6, 11, 19]  # Indices of informative features

    # Generate synthetic data with make_classification
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=3,
        n_redundant=0,
        n_repeated=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=2024,
    )

    return X, y, informative_features


@pytest.fixture
def fake_data_regression():
    X, y = make_regression(
        n_samples=1000, n_features=100, n_informative=2, random_state=1
    )
    informative_features = [85, 32]
    return X, y, informative_features


def test_fake_data_classification(fake_data_classification):
    X, y, expected_informative_features = fake_data_classification
    assert len(expected_informative_features) == 3
    assert X.shape[0] == 1000
    assert X.shape[1] == 20
    assert all(idx in range(X.shape[1]) for idx in expected_informative_features)


def test_fake_data_regression(fake_data_regression):
    X, y, expected_informative_features = fake_data_regression
    assert len(expected_informative_features) == 2
    assert X.shape[0] == 1000
    assert X.shape[1] == 100
    assert all(idx in range(X.shape[1]) for idx in expected_informative_features)


def test_feature_selection_f_statistic_regression(fake_data_regression):
    X, y, expected_features = fake_data_regression
    selector = FStatisticSelector(task="regression", num_features_to_select=2)
    scores, selected_features = selector.select_features(X, y)
    assert len(scores) == 100
    assert len(selected_features) == 2
    assert set(selected_features) == set(expected_features)


def test_feature_selection_f_statistic_classification(fake_data_classification):
    X, y, expected_features = fake_data_classification
    selector = FStatisticSelector(task="classification", num_features_to_select=3)
    scores, selected_features = selector.select_features(X, y)
    assert len(scores) == 20
    assert len(selected_features) == 3
    assert set(selected_features) == set(expected_features)


def test_feature_selection_mutual_info_classification(fake_data_classification):
    X, y, expected_features = fake_data_classification
    selector = MutualInfoSelector(task="classification", num_features_to_select=3)
    scores, selected_features = selector.select_features(X, y)
    assert len(scores) == 20
    assert len(selected_features) == 3
    assert set(selected_features) == set(expected_features)


def test_feature_selection_mutual_info_regression(fake_data_regression):
    X, y, expected_features = fake_data_regression
    selector = MutualInfoSelector(task="regression", num_features_to_select=2)
    scores, selected_features = selector.select_features(X, y)
    assert len(scores) == 100
    assert len(selected_features) == 2
    assert set(selected_features) == set(expected_features)


def test_feature_selection_xgboost_classification(fake_data_classification):
    X, y, expected_features = fake_data_classification
    selector = XGBoostSelector(task="classification", num_features_to_select=3)
    scores, selected_features = selector.select_features(X, y)
    assert len(scores) == 20
    assert len(selected_features) == 3
    assert set(selected_features) == set(expected_features)


def test_feature_selection_xgboost_regression(fake_data_regression):
    X, y, expected_features = fake_data_regression
    selector = XGBoostSelector(task="regression", num_features_to_select=2)
    scores, selected_features = selector.select_features(X, y)
    assert len(scores) == 100
    assert len(selected_features) == 2
    assert set(selected_features) == set(expected_features)


def test_feature_selection_random_forest_classification(fake_data_classification):
    X, y, expected_features = fake_data_classification
    selector = RandomForestSelector(task="classification", num_features_to_select=3)
    scores, selected_features = selector.select_features(X, y)
    assert len(scores) == 20
    assert len(selected_features) == 3
    assert set(selected_features) == set(expected_features)


def test_feature_selection_random_forest_regression(fake_data_regression):
    X, y, expected_features = fake_data_regression
    selector = RandomForestSelector(task="regression", num_features_to_select=2)
    scores, selected_features = selector.select_features(X, y)
    assert len(scores) == 100
    assert len(selected_features) == 2
    assert set(selected_features) == set(expected_features)


def test_feature_selection_svm_classification(fake_data_classification):
    X, y, expected_features = fake_data_classification
    selector = SVMSelector(task="classification", num_features_to_select=3)
    scores, selected_features = selector.select_features(X, y)
    assert len(scores) == 20
    assert len(selected_features) == 3
    assert set(selected_features) == set(expected_features)


def test_feature_selection_svm_regression(fake_data_regression):
    X, y, expected_features = fake_data_regression
    selector = SVMSelector(task="regression", num_features_to_select=2)
    scores, selected_features = selector.select_features(X, y)
    assert len(scores) == 100
    assert len(selected_features) == 2
    assert set(selected_features) == set(expected_features)


def test_feature_selection_mrmr_classification(fake_data_classification):
    X, y, expected_features = fake_data_classification
    selector = MRMRSelector(task="classification", num_features_to_select=3)
    scores, selected_features = selector.select_features(X, y)
    assert len(scores) == 20
    assert len(selected_features) == 3
    assert set(selected_features) == set(expected_features)


def test_feature_selection_mrrm_regression(fake_data_regression):
    X, y, expected_features = fake_data_regression
    selector = MRMRSelector(task="regression", num_features_to_select=2)
    scores, selected_features = selector.select_features(X, y)
    assert len(scores) == 100
    assert len(selected_features) == 2
    assert set(selected_features) == set(expected_features)


def test_feature_selection_lasso_classification(fake_data_classification):
    X, y, expected_features = fake_data_classification
    selector = LassoSelector(task="classification", num_features_to_select=3)
    scores, selected_features = selector.select_features(X, y)
    assert len(scores) == 20
    assert len(selected_features) == 3
    assert set(selected_features) == set(expected_features)


def test_feature_selection_lasso_regression(fake_data_regression):
    X, y, expected_features = fake_data_regression
    selector = LassoSelector(task="regression", num_features_to_select=2)
    scores, selected_features = selector.select_features(X, y)
    assert len(scores) == 100
    assert len(selected_features) == 2
    assert set(selected_features) == set(expected_features)
