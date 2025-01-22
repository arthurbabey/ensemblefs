import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import pytest

from ensemblefs.feature_selection_pipeline import FeatureSelectionPipeline


def test_expected_features():
    """
    Test whether the pipeline can identify at least 90% of the informative features
    among the top 30 selected features in a synthetic dataset.
    """
    # Parameters for synthetic data
    num_samples = 1000
    num_features = 20
    num_informative = 3
    n_redundant = 0
    n_repeated = 0
    n_classes = 3
    n_clusters_per_class = 1
    random_state = 2024

    # Generate synthetic classification data
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=num_samples,
        n_features=num_features,
        n_informative=num_informative,
        n_redundant=n_redundant,
        n_repeated=n_repeated,
        n_classes=n_classes,
        n_clusters_per_class=n_clusters_per_class,
        random_state=random_state,
    )

    # Convert to pandas DataFrame
    feature_names = [f"Feature_{i+1}" for i in range(num_features)]
    data = pd.DataFrame(X, columns=feature_names)
    data["target"] = y

    # Standard parameters for the pipeline
    fs_methods = [
        "f_statistic_selector",
        "random_forest_selector",
        "mutual_info_selector",
        "xgboost_selector",
        "svm_selector",
    ]
    merging_strategy = "union_of_intersections_merger"
    num_repeats = 5
    num_features_to_select = 15
    task = "classification"
    metrics = ["accuracy", "f1_score"]

    # Initialize the pipeline
    pipeline = FeatureSelectionPipeline(
        data=data,
        fs_methods=fs_methods,
        merging_strategy=merging_strategy,
        num_repeats=num_repeats,
        num_features_to_select=num_features_to_select,
        task=task,
        metrics=metrics,
        random_state=random_state,
        n_jobs=10,
    )

    # Run the pipeline
    selected_features, _, _ = pipeline.run()
    print(len(selected_features))

    # Extract the informative feature names
    informative_features = set(
        feature_names[:num_informative]
    )  # First `num_informative` features

    # Compute the overlap between selected and informative features
    selected_informative = set(selected_features) & informative_features
    proportion_found = len(selected_informative) / len(informative_features)

    # Assert at least 90% of informative features were found
    assert proportion_found >= 0.9, (
        f"Expected at least 90% of informative features to be selected, "
        f"but only {proportion_found * 100:.2f}% were found."
    )
