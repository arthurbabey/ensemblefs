import argparse
import csv
import os
import shutil
from pathlib import Path

import pandas as pd
import yaml

from ensemblefs import FeatureSelectionPipeline
from ensemblefs.core import DataProcessor
from scripts.utils import (
    read_config,
    validate_experience_config,
    validate_pipeline_config,
    validate_preprocessing_config,
)


def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def main():
    parser = argparse.ArgumentParser(description="Run the main application script.")

    # Argument for the configuration file
    parser.add_argument(
        "--config",  # Use -- to indicate this is a flag (optional argument)
        nargs="?",  # Keep it optional with nargs="?"
        default=os.path.join(
            Path(__file__).parent, "config.yml"
        ),  # Default to packaged scripts/config.yml
        help="Path to the configuration file. Defaults to the packaged config.yml.",
    )

    # Argument for the processed data path (optional)
    parser.add_argument(
        "--data",  # Use -- to indicate this is a flag (optional argument)
        nargs="?",  # Keep it optional with nargs="?"
        default=None,  # Default to None if not provided
        help="Path to the processed data file. If not provided, a raw dataset should be defined in the config file.",
    )

    # Parse arguments
    args = parser.parse_args()
    config_path = args.config
    dataset = args.data

    # Load the configuration
    config = load_config(config_path)
    # print(f"Configuration loaded: {config}")

    # Validate configuration files
    config_experience, config_preprocessing, config_pipeline = read_config(config)
    validate_pipeline_config(config_pipeline)

    if config_preprocessing:
        validate_preprocessing_config(config_preprocessing)

    if config_experience:
        validate_experience_config(config_experience)

    if dataset is None:
        data_path = config_experience.get("data_path", {}).get("value", None)
        if data_path is None:
            raise ValueError(
                "No data path provided. Please specify a processed data file or a raw data path in the config."
            )

    metadata_path = config_experience.get("metadata_path", {}).get("value", None)
    result_path = config_experience.get("result_path", {}).get(
        "value", "default_results"
    )
    experiment_name = config_experience.get("experiment_name", {}).get(
        "value", "default_experiment"
    )
    experiment_folder = os.path.join(result_path, experiment_name)
    os.makedirs(experiment_folder, exist_ok=True)

    shutil.copy(config_path, experiment_folder)

    # Extract preprocessing parameters from configuration
    categorical_columns = config_preprocessing.get("categorical_columns", None)
    columns_to_drop = config_preprocessing.get("columns_to_drop", None)
    drop_missing_values = config_preprocessing.get("drop_missing_values", False)
    merge_key = config_preprocessing.get("merge_key", None)
    normalize = config_preprocessing.get("normalize", True)
    target_column = config_preprocessing.get("target_column", "target")

    if args.data is None:
        data_processor = DataProcessor(
            categorical_columns=categorical_columns,
            columns_to_drop=columns_to_drop,
            drop_missing_values=drop_missing_values,
            merge_key=merge_key,
            normalize=normalize,
            target_column=target_column,
        )
        dataset = data_processor.preprocess_data(data=data_path, metadata=metadata_path)
    else:
        dataset = pd.read_csv(args.data)

    # Extract pipeline parameters
    fs_methods = config_pipeline["fs_methods"]["value"]
    merging_strategy = config_pipeline["merging_strategy"]["value"]
    num_repeats = config_pipeline["num_repeats"]["value"]
    num_features_to_select = config_pipeline["num_features_to_select"]["value"]
    metrics = config_pipeline["metrics"]["value"]
    task = config_pipeline["task"]["value"]
    random_state = config_pipeline.get("random_state", {}).get("value", None)
    n_jobs = config_pipeline.get("n_jobs", {}).get("value", None)

    # Run pipeline
    pipeline = FeatureSelectionPipeline(
        data=dataset,
        fs_methods=fs_methods,
        merging_strategy=merging_strategy,
        num_repeats=num_repeats,
        num_features_to_select=num_features_to_select,
        # metrics=metrics,
        task=task,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    best_features, best_repeat, best_group_name = pipeline.run(verbose=False)

    # Save results in a text file
    with open(os.path.join(experiment_folder, "results.txt"), "w") as file:
        file.write(f"The best features are {best_features}\n")
        file.write(f"Best repeat value: {best_repeat}\n")
        file.write(f"Best group name: {best_group_name}\n")

    # Save results in a CSV file
    results = {
        "best_features": best_features,
        "best_repeat": best_repeat,
        "best_group_name": best_group_name,
    }

    csv_file_path = os.path.join(experiment_folder, "results.csv")
    with open(csv_file_path, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=results.keys())
        writer.writeheader()
        writer.writerow(results)
        print(f"Results written to {experiment_folder}")


if __name__ == "__main__":
    main()
