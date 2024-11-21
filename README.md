# ensemblefs [![tests](https://github.com/arthurbabey/ensemblefs/actions/workflows/tests.yml/badge.svg)](https://github.com/arthurbabey/ensemblefs/actions/workflows/tests.yml) [![Documentation](https://github.com/arthurbabey/ensemblefs/actions/workflows/doc.yml/badge.svg)](https://github.com/arthurbabey/ensemblefs/actions/workflows/doc.yml)

* [Documentation](https://arthurbabey.github.io/ensemblefs/)

## Overview

The **ensemblefs package** is a tool designed to automate feature selection with multi-objective optimization, using an ensemble approach. It combines multiple feature selection methods, assesses each combination across several repetitions, and uses Pareto optimization to balance performance and stability. You can use this pipeline as a Python library or interact through a command-line interface (CLI) with `efs-pipeline`, which runs the main script `scripts/main.py`.

---

## Requirements

- **Python** 3.9 or higher
- **Dependencies**: All required packages are listed in the `pyproject.toml` file and will be installed automatically.

---

## Installation

### From Source

To install the package from source, follow these steps:

1. Install the package directly from the repository:

   ```bash
   pip install git+https://github.com/arthurbabey/ensemblefs.git
   ```

2. Alternatively, clone the repository and install locally:

   ```bash
   git clone https://github.com/arthurbabey/ensemblefs.git
   cd ensemblefs
   pip install .
   ```

   Or, if you are planning to modify the source code and want changes to be reflected immediately without reinstalling, you can install the package in **editable mode**:

   ```bash
   pip install -e .
   ```

In **editable mode**, any modifications to the source code will take effect immediately, making it ideal for development.

---

> **Note**: This project will soon be available on [PyPI](https://pypi.org/), allowing for easy installation with:

   ```bash
   pip install ensemblefs
   ```

---

## Code Structure

- **`core/`**: Contains core modules for data processing, metrics, and algorithm-specific logic.
- **`feature_selection_pipeline.py`**: Defines the main feature selection pipeline logic. The `.run()` method is the primary entry point.
- **`feature_selectors/`**: Defines feature selection methods like F-statistic, mutual information, RandomForest, and SVM.
- **`merging_strategies/`**: Contains merging strategies such as Borda count and union of intersections.

---

## Using the Package

### 1. Using the Library

To use as a library, import and instantiate the main functionality in `ensemblefs/feature_selection_pipeline.py`. The `.run()` method executes the pipeline using specific parameters. Refer to the [tutorials](https://github.com/arthurbabey/ensemblefs/tree/main/tutorials) or [documentation](https://arthurbabey.github.io/ensemblefs/) for parameter details.

```python
from ensemblefs import FeatureSelectionPipeline

# Example usage
fs_methods = [
    "f_statistic_selector",
    "random_forest_selector",
    "svm_selector"
]
merging_strategy = "union_of_intersections_merger"
pipeline = FeatureSelectionPipeline(
    data=data,
    fs_methods=fs_methods,
    merging_strategy=merging_strategy,
    num_repeats=5,
    task="classification",
    num_features_to_select=10
)
results = pipeline.run()
```

---

### 2. Using the Command Line Interface (CLI)

Once the package is installed, you can run the pipeline using the `efs-pipeline` command from the command line. This command executes `scripts/main.py`, which utilizes the parameters defined in the `scripts/config.yaml` file. This configuration file contains **all the necessary settings for the pipeline**, including experiment details, preprocessing options, and feature selection parameters. It shows valid values and a function validate your values before running the experiment.

```bash
efs-pipeline
```

By default, the script will use the `config.yaml` file located in the `scripts/` directory. However, if you wish to override the default configuration, you can copy `scripts/config.yaml` to your working directory, modify it as needed, and specify your custom configuration file:

```bash
efs-pipeline path/to/your_config.yaml
```

#### Configuration Details

The pipeline relies primarily on the **pipeline** section of the `config.yaml` file. While the **experiment** and **preprocessing** sections are optional, including them helps to better organize and manage the pipeline execution.

- The **experiment** section specifies the experiment name, paths to the raw data, and the results directory.
- The **preprocessing** section defines preprocessing steps like normalization or missing value handling.
- The **pipeline** section configures the feature selection methods, merging strategy, task type, and other parameters for the pipeline run.

If you don’t include the experiment section at all, the pipeline will still function, but you must explicitly provide a processed dataset as a second argument. For example, in the following command, `path/to/your_config.yaml` points to your configuration file, and `/path/to/processed_data.csv` is the preprocessed dataset:

```bash
efs-pipeline path/to/your_config.yaml /path/to/processed_data.csv
```

#### Example `config.yaml`

Here is an example of how a `config.yaml` file may look. The **pipeline** section is required, while **experiment** and **preprocessing** are optional but recommended for easier management:

```yaml
experiment:
  name: "example_experiment"
  results_path: "results/"
  data_path: "data/input_data.csv"

preprocessing:
  normalize: true
  handle_missing: true

pipeline:
  fs_methods: ["f_statistic_selector", "random_forest_selector"]
  merging_strategy: "union_of_intersections_merger"
  num_repeats: 5
  task: "classification"
  num_features_to_select: 10
```

#### Results

Upon execution, the script creates a `results` directory at the location specified in `config.yaml` under `results_path`, organized by the experiment name. For example, results will be stored at:

```
results/example_experiment/
```

Within this directory, the pipeline will save two files:

- A **text file** containing a summary of the pipeline run.
- A **CSV file** containing the final results.

By adding the **experiment** and **preprocessing sections**, you can better manage multiple experiments and reuse common preprocessing steps across different runs.
