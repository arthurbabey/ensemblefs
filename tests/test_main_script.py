import os
import subprocess
import sys

import pandas as pd
import pytest

# Define the path to the script and config
SCRIPT_PATH = os.path.join("scripts", "main.py")
CONFIG_PATH = os.path.join("tests", "config_test.yml")
DATASET_PATH = os.path.join("tests", "test_data.csv")


@pytest.fixture
def sample_data():
    data = {
        "column1": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
        "column2": ["X", "Y", "X", "Y", "X", "Y", "X", "Y", "X", "Y", "X", "Y"],
        "column3": [1, 21, 3, 4, 5, 6, 7, 3, 81, 9, 112, 1],
        "column4": [1, 2, 333, 4, 5, 61, 7, 3, 8, 9, 10, 121],
        "column5": [11, 21, 31, 4, 5, 6, 7, 3, 8, 9, 10, 11],
        "column6": [1, 2, 3, 4, 51, 6, 7, 31, 8, 9, 55, 11],
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    }
    df = pd.DataFrame(data)
    df.to_csv(DATASET_PATH, index=False)
    return df


def test_main_script_with_config(sample_data):
    result = subprocess.run(
        [sys.executable, SCRIPT_PATH, "--config", CONFIG_PATH],
        capture_output=True,
        text=True,
    )
    os.remove(DATASET_PATH)
    os.remove("tests/test_experiment/results.txt")
    os.remove("tests/test_experiment/results.csv")
    os.remove("tests/test_experiment/config_test.yml")
    os.removedirs("tests/test_experiment")
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"


def test_command_line_interface_with_config(sample_data):
    result = subprocess.run(
        [sys.executable, SCRIPT_PATH, "--config", CONFIG_PATH],
        capture_output=True,
        text=True,
    )
    os.remove(DATASET_PATH)
    os.remove("tests/test_experiment/results.txt")
    os.remove("tests/test_experiment/results.csv")
    os.remove("tests/test_experiment/config_test.yml")
    os.removedirs("tests/test_experiment")
    assert result.returncode == 0, f"CLI command failed with error: {result.stderr}"


def test_command_line_interface_with_dataset(sample_data):
    # read and process sample_data
    data = pd.read_csv(DATASET_PATH)
    data["column1"] = data["column1"].astype("category").cat.codes
    data["column2"] = data["column2"].astype("category").cat.codes
    data.to_csv(DATASET_PATH, index=False)

    result = subprocess.run(
        [sys.executable, SCRIPT_PATH, "--config", CONFIG_PATH, "--data", DATASET_PATH],
        capture_output=True,
        text=True,
    )
    os.remove(DATASET_PATH)
    os.remove("tests/test_experiment/results.txt")
    os.remove("tests/test_experiment/results.csv")
    os.remove("tests/test_experiment/config_test.yml")
    os.removedirs("tests/test_experiment")
    assert result.returncode == 0, f"CLI command failed with error: {result.stderr}"
