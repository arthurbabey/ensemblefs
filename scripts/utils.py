from typing import Any, Dict, List, Tuple


def read_config(config: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Read configuration sections from the provided config dictionary.

    Args:
        config: The configuration dictionary.

    Returns:
        A tuple containing the 'experience', 'preprocessing', and 'pipeline' sections.

    Raises:
        ValueError: If the 'pipeline' section is missing.
    """
    config_pipeline = config.get("pipeline", {})
    config_experience = config.get("experience", {})
    config_preprocessing = config.get("preprocessing", {})

    if not config_pipeline:
        raise ValueError("The configuration file must contain a 'pipeline' section.")

    return config_experience, config_preprocessing, config_pipeline


def validate_pipeline_config(config: Dict[str, Any]) -> None:
    """
    Validate the pipeline configuration.

    Args:
        config: The pipeline configuration dictionary.

    Raises:
        ValueError: If any required pipeline parameter is invalid.
    """
    # Validate fs_methods
    fs_methods: List[Any] = config.get("fs_methods", {}).get("value", [])
    valid_fs_methods = {
        "f_statistic_selector",
        "mutual_info_selector",
        "random_forest_selector",
        "svm_selector",
        "xgboost_selector",
        "rfe_rf_selector",
    }
    if not isinstance(fs_methods, list) or len(fs_methods) < 2:
        raise ValueError("`fs_methods` should be a list containing two or more feature selection methods.")
    if any(method not in valid_fs_methods for method in fs_methods):
        raise ValueError(
            f"Invalid feature selection method(s) in `fs_methods`: {fs_methods}. Valid options are: {valid_fs_methods}."
        )

    # Validate merging_strategy
    merging_strategy: str = config.get("merging_strategy", {}).get("value", "")
    valid_merging_strategies = {
        "union_of_intersections_merger",
        "borda_merger",
    }
    if merging_strategy not in valid_merging_strategies:
        raise ValueError(
            f"Invalid `merging_strategy`: {merging_strategy}. Choose one from: {valid_merging_strategies}."
        )

    # Validate num_repeats
    num_repeats = config.get("num_repeats", {}).get("value", None)
    if not isinstance(num_repeats, int) or not (2 <= num_repeats <= 10):
        raise ValueError(f"`num_repeats` must be an integer between 2 and 10, inclusive. Got: {num_repeats}.")

    # Validate random_state
    random_state = config.get("random_state", {}).get("value", None)
    if not (isinstance(random_state, int) or random_state is None):
        raise ValueError("`random_state` must be an integer or None.")

    # Validate metrics
    metrics: List[Any] = config.get("metrics", {}).get("value", [])
    valid_metrics = {
        "accuracy",
        "logloss",
        "f1_score",
        "precision_score",
        "recall_score",
        "mse",
        "mae",
        "r2_score",
    }
    if not isinstance(metrics, list) or not all(metric in valid_metrics for metric in metrics):
        raise ValueError(
            f"`metrics` should be a list containing metric names from the following options: {valid_metrics}."
        )

    # Validate task
    task: str = config.get("task", {}).get("value", "")
    valid_tasks = {"regression", "classification"}
    if task not in valid_tasks:
        raise ValueError(f"Invalid `task`: {task}. Choose either 'regression' or 'classification'.")

    # Validate num_features_to_select
    num_features_to_select = config.get("num_features_to_select", {}).get("value", None)
    if not (isinstance(num_features_to_select, int) or num_features_to_select is None):
        raise ValueError("`num_features_to_select` must be an integer or None.")

    # Validate n_jobs
    n_jobs = config.get("n_jobs", {}).get("value", None)
    if not (isinstance(n_jobs, int) and (n_jobs > 0 or n_jobs == -1) or n_jobs is None):
        raise ValueError("`n_jobs` must be a positive integer, -1, or None.")

    print("Configuration is valid.")


def validate_preprocessing_config(config: Dict[str, Any]) -> None:
    """
    Validate the preprocessing configuration.

    Args:
        config: The preprocessing configuration dictionary.

    Raises:
        ValueError: If any preprocessing parameter is invalid.
    """
    # Validate categorical_columns
    categorical_columns = config.get("categorical_columns", [])
    if not isinstance(categorical_columns, list) or not all(isinstance(col, str) for col in categorical_columns):
        raise ValueError("`categorical_columns` should be a list of strings representing column names.")

    # Validate columns_to_drop
    columns_to_drop = config.get("columns_to_drop", [])
    if not isinstance(columns_to_drop, list) or not all(isinstance(col, str) for col in columns_to_drop):
        raise ValueError("`columns_to_drop` should be a list of strings representing column names.")

    # Validate drop_missing_values
    drop_missing_values = config.get("drop_missing_values", None)
    if not isinstance(drop_missing_values, bool):
        raise ValueError("`drop_missing_values` must be a boolean (True or False).")

    # Validate merge_key
    merge_key = config.get("merge_key", "")
    if not isinstance(merge_key, str):
        raise ValueError("`merge_key` should be a string representing a column name.")

    # Validate normalize
    normalize = config.get("normalize", None)
    if not isinstance(normalize, bool):
        raise ValueError("`normalize` must be a boolean (True or False).")

    # Validate target_column
    target_column = config.get("target_column", "")
    if not isinstance(target_column, str):
        raise ValueError("`target_column` should be a string representing a column name.")

    print("Preprocessing configuration is valid.")


def validate_experience_config(config: Dict[str, Any]) -> None:
    """
    Validate the experience configuration.

    Args:
        config: The experience configuration dictionary.

    Raises:
        ValueError: If any experience parameter is invalid.
    """
    # Validate data_path
    data_path = config.get("data_path", {}).get("value", "")
    if not isinstance(data_path, str) or not data_path:
        raise ValueError("`data_path` must be a non-empty string representing the path to the dataset.")

    # Validate experiment_name
    experiment_name = config.get("experiment_name", {}).get("value", "")
    if not isinstance(experiment_name, str) or not experiment_name:
        raise ValueError(
            "`experiment_name` must be a non-empty string representing the name for the experiment results."
        )

    # Validate metadata_path
    metadata_path = config.get("metadata_path", {}).get("value", "")
    if not (isinstance(metadata_path, str) or metadata_path is None):
        raise ValueError("`metadata_path` must be a string representing the path to the metadata file or None.")

    # Validate result_path
    result_path = config.get("result_path", {}).get("value", "")
    if not isinstance(result_path, str) or not result_path:
        raise ValueError("`result_path` must be a non-empty string representing the directory path to save results.")

    print("Experience configuration is valid.")
