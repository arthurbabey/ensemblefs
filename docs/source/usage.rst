Usage
=====

Overview
--------

The **ensemblefs** package provides a customizable, ensemble-based feature selection method using multi-objective optimization to identify the most stable and high-performing feature subsets from a dataset. The core of this functionality is encapsulated within the `FeatureSelectionPipeline` class, which orchestrates the entire selection process.

Getting Started
---------------

To start using the **ensemblefs** package, you first need to import the main pipeline class and configure it with the feature selection methods and merging strategy of your choice.

.. code-block:: python

    from ensemblefs import FeatureSelectionPipeline

Configuring the Pipeline
------------------------

The `FeatureSelectionPipeline` allows for flexible configuration with various built-in feature selection methods, as well as custom methods and merging strategies. Here's an example of setting up a pipeline:

.. code-block:: python

    # Define your feature selectors
    from ensemblefs.feature_selectors import RandomForestSelector, MutualInfoSelector

    # Import the merger strategy
    from ensemblefs.merging_strategies import BordaMerger

    # Create an instance of the pipeline
    pipeline = FeatureSelectionPipeline(
        fs_methods=[RandomForestSelector(), MutualInfoSelector()],
        merger=BordaMerger(),
        min_fs_methods=2,
        max_fs_methods=5,
        repeats=10
    )

Running the Pipeline
--------------------

Once configured, you can run the pipeline on your dataset to perform feature selection. The pipeline evaluates combinations of feature selectors, computes performance and stability metrics, and employs Pareto dominance to determine optimal feature subsets.

.. code-block:: python

    # Assuming 'X' and 'y' are your features and target variable respectively
    best_features = pipeline.run_pipeline(X, y)
    print("Selected features:", best_features)

This example demonstrates the use of RandomForest and Mutual Information selectors with a Borda merger strategy, but you can easily substitute these with any other built-in selectors or custom ones derived from the provided base classes.

Extendability
-------------

You can also extend the functionality by creating custom feature selectors,  mergers and metrics. Inherit from the base classes found under `feature_selectors` and `merging_strategies` or `base_metrics` directories and implement the necessary methods as per your requirements.

.. code-block:: python

    from ensemblefs.feature_selectors.base_selector import BaseSelector

    class CustomSelector(BaseSelector):
        def select_features(self, X, y):
            # Implement your feature selection logic here
            pass
