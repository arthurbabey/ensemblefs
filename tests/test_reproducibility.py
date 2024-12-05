import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ensemblefs.feature_selection_pipeline import FeatureSelectionPipeline


# Fixture with dynamic task and metrics
@pytest.fixture(
    params=[
        (
            "classification",
            "union_of_intersections_merger",
            ["logloss", "f1_score", "accuracy"],
        ),
        (
            "classification",
            "borda_merger",
            ["precision_score", "recall_score", "accuracy"],
        ),
        ("regression", "union_of_intersections_merger", ["mse", "mae", "r2_score"]),
        ("regression", "borda_merger", ["mae", "r2_score", "mse"]),
    ]
)
def pipeline_instance(request):
    """
    Fixture to generate pipeline instances with different configurations.
    """
    num_samples = 50
    num_features = 25
    feature_names = [f"Feature_{i+1}" for i in range(num_features)]
    target_values_classification = np.random.randint(0, 4, size=num_samples)
    target_values_regression = np.random.randn(num_samples)

    data = pd.DataFrame(
        np.random.randn(num_samples, num_features), columns=feature_names
    )
    task, merging_strategy, metrics = request.param

    if task == "classification":
        data["target"] = target_values_classification
    else:
        data["target"] = target_values_regression

    fs_methods = [
        "f_statistic_selector",
        "random_forest_selector",
        "mutual_info_selector",
    ]
    num_repeats = 2
    random_state = 2024
    num_features_to_select = 10
    n_jobs = 10

    pipeline = FeatureSelectionPipeline(
        data=data,
        fs_methods=fs_methods,
        merging_strategy=merging_strategy,
        num_repeats=num_repeats,
        num_features_to_select=num_features_to_select,
        task=task,
        metrics=metrics,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    return pipeline


# Test reproducibility
def test_pipeline_reproducibility(pipeline_instance):
    """
    Run the pipeline multiple times and verify that the results are consistent.
    """
    # Run the pipeline for the first time
    first_run = pipeline_instance.run()
    outcomes = [first_run]

    # Run the pipeline multiple times
    for _ in range(3):
        outcome = pipeline_instance.run()
        outcomes.append(outcome)

    # Compare all outcomes to the first run
    for i in range(1, len(outcomes)):
        assert (
            outcomes[i] == outcomes[0]
        ), f"Pipeline results are not reproducible: Run {i} differs."


def test_pipeline_reproducibility_across_instances():
    """
    Create multiple pipeline instances with the same parameters and verify that
    their results are consistent when run independently.
    """
    # Define parameters
    num_samples = 50
    num_features = 25
    feature_names = [f"Feature_{i+1}" for i in range(num_features)]
    data = pd.DataFrame(
        np.random.randn(num_samples, num_features), columns=feature_names
    )
    data["target"] = np.random.randint(0, 4, size=num_samples)  # Classification task

    fs_methods = [
        "f_statistic_selector",
        "random_forest_selector",
        "mutual_info_selector",
    ]
    merging_strategy = "union_of_intersections_merger"
    num_repeats = 2
    random_state = 2024
    num_features_to_select = 10
    task = "classification"
    metrics = ["logloss", "f1_score", "accuracy"]
    n_jobs = 10

    # Create N pipeline instances
    num_instances = 5
    pipelines = [
        FeatureSelectionPipeline(
            data=data.copy(),
            fs_methods=fs_methods,
            merging_strategy=merging_strategy,
            num_repeats=num_repeats,
            num_features_to_select=num_features_to_select,
            task=task,
            metrics=metrics,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        for _ in range(num_instances)
    ]

    # Run each pipeline and collect results
    outcomes = [pipeline.run() for pipeline in pipelines]

    # Check if all outcomes are identical
    first_outcome = outcomes[0]
    for i, outcome in enumerate(outcomes[1:], start=1):
        assert (
            outcome == first_outcome
        ), f"Pipeline results are not reproducible: Instance {i} differs."
