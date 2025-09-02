import os
import sys

import numpy as np
import pandas as pd
import pytest

# Ensure the project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ensemblefs.feature_selection_pipeline import FeatureSelectionPipeline


@pytest.fixture
def pipeline_args():
    # Deterministic data generation
    num_samples = 50
    num_features = 25
    np.random.seed(0)
    feature_names = [f"Feature_{i+1}" for i in range(num_features)]
    data = pd.DataFrame(
        np.random.randn(num_samples, num_features), columns=feature_names
    )
    data["target"] = np.random.randint(0, 4, size=num_samples)  # Classification task

    fs_methods = [
        "f_statistic_selector",
        "random_forest_selector",
        "mutual_info_selector",
        "lasso_selector",
        "xgboost_selector",
        "svm_selector",
    ]
    merging_strategy = "union_of_intersections_merger"
    num_repeats = 3
    random_state = 2024
    num_features_to_select = 10
    task = "classification"
    metrics = ["logloss", "f1_score", "accuracy"]
    n_jobs = 3  # Test with parallelism

    return {
        "data": data.copy(),
        "fs_methods": fs_methods,
        "merging_strategy": merging_strategy,
        "num_repeats": num_repeats,
        "num_features_to_select": num_features_to_select,
        "task": task,
        "metrics": metrics,
        "random_state": random_state,
        "n_jobs": n_jobs,
    }


def test_pipeline_reproducibility_different_runs(pipeline_args):
    # Create a pipeline instance
    pipeline = FeatureSelectionPipeline(**pipeline_args)

    # Run the pipeline multiple times on the same instance
    result1 = pipeline.run()
    result2 = pipeline.run()
    result3 = pipeline.run()

    assert (
        result1 == result2 == result3
    ), "Multiple runs on the same instance yield different results"


def test_pipeline_reproducibility_different_instances(pipeline_args):
    # Create two separate pipeline instances with identical parameters
    pipeline1 = FeatureSelectionPipeline(**pipeline_args)
    pipeline2 = FeatureSelectionPipeline(**pipeline_args)
    pipeline3 = FeatureSelectionPipeline(**pipeline_args)

    result1 = pipeline1.run()
    result2 = pipeline2.run()
    result3 = pipeline3.run()
    assert result1 == result2 == result3, "Different instances yield different results"
