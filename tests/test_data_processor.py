import os
import sys

import pandas as pd
import pytest

from ensemblefs.core.data_processor import DataProcessor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture
def data_processor():
    """
    Fixture to initialize the DataProcessor with a test configuration.
    """
    config = {
        "categorical_columns": ["category"],
        "columns_to_drop": ["unnecessary_column"],
        "drop_missing_values": True,
        "merge_key": "id",
        "normalize": True,
        "target_column": "target",
    }
    return DataProcessor(**config)


def create_dummy_csv(data, filename):
    data_df = pd.DataFrame(data)
    data_df.to_csv(filename, index=False)


def cleanup_file(filename):
    if os.path.exists(filename):
        os.remove(filename)


def test_preprocess_data(data_processor):
    # Dummy dataset
    data = {
        "id": [1, 2, 3],
        "category": ["A", "B", "C"],
        "value": [10, 20, 30],
        "unnecessary_column": [0, 0, 0],
        "target": [1, 0, 1],
    }
    filename = "dummy_data.csv"
    create_dummy_csv(data, filename)

    # Test preprocessing
    processed_data = data_processor.preprocess_data(filename)
    assert "unnecessary_column" not in processed_data.columns
    assert processed_data["category"].dtype in ["int32", "int64"]
    assert "target" in processed_data.columns
    assert abs(processed_data["value"].mean()) <= 1e-15
    cleanup_file(filename)


def test_drop_missing_values(data_processor):
    # Dummy dataset with missing values
    data_with_missing = {
        "id": [1, 2, 3],
        "category": ["A", None, "C"],
        "value": [10, 20, None],
        "target": [1, 0, 1],
    }
    filename = "missing_data.csv"
    create_dummy_csv(data_with_missing, filename)
    processed_data = data_processor.preprocess_data(filename)
    assert len(processed_data) == 1  # Only the last row has complete data
    cleanup_file(filename)
