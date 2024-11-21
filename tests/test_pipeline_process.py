import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import pytest

from ensemblefs.feature_selection_pipeline import FeatureSelectionPipeline


@pytest.fixture(params=["union_of_intersections_merger", "borda_merger"])
def pipeline_instance(request):

    num_samples = 100
    num_features = 5000
    feature_names = [f"Feature_{i+1}" for i in range(num_features)]
    target_values = np.random.randint(0, 4, size=num_samples)

    data = pd.DataFrame(
        np.random.randn(num_samples, num_features), columns=feature_names
    )
    data["target"] = target_values

    fs_methods = [
        "f_statistic_selector",
        "random_forest_selector",
        "mutual_info_selector",
    ]
    merging_strategy = request.param
    num_repeats = 2
    task = "classification"
    random_state = 2024
    num_features_to_select = 300
    n_jobs = 1
    pipeline = FeatureSelectionPipeline(
        data=data,
        fs_methods=fs_methods,
        merging_strategy=merging_strategy,
        num_repeats=num_repeats,
        num_features_to_select=num_features_to_select,
        task=task,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    return pipeline


def test_feature_selection_pipeline(pipeline_instance):
    (
        best_features,
        best_repeat,
        best_group_name,
    ) = pipeline_instance()  # or pipeline_instance.run()
    assert best_features is not None
    assert best_repeat is not None
    assert best_group_name is not None

    # test best_features are in the data columns
    assert all([feature in pipeline_instance.data.columns for feature in best_features])
    assert 0 <= int(best_repeat) <= pipeline_instance.num_repeats
    assert best_group_name in pipeline_instance.subgroup_names
