import numpy as np


class FeatureSelector:
    """
    Base class for feature selection. This class provides the interface and common functionality for selecting features based on their scores.

    Attributes:
        task (str): Specifies the machine learning task, either 'classification' or 'regression'.
        num_features_to_select (int): The number of top features to select based on importance.
            If None, a default selection logic can be applied based on a percentage of features.
    """

    def __init__(self, task=None, num_features_to_select=None):
        """
        Initializes the FeatureSelector with the specified task and number of features.

        Args:
            task (str, optional): The machine learning task ("classification" or "regression"). Defaults to None.
            num_features_to_select (int, optional): The number of features to select. If None, selection defaults to 10% of features.
        """
        self.task = task
        self.num_features_to_select = num_features_to_select

    def select_features(self, X, y):
        """
        Selects the top features based on their computed scores.

        Args:
            X (array-like, shape = [n_samples, n_features]): Training input samples.
            y (array-like, shape = [n_samples] or [n_samples, n_outputs]): Target values (class labels for classification, real numbers for regression).

        Returns:
            tuple: A tuple containing the feature scores and the indices of the selected features.
        """
        if self.num_features_to_select is None:
            self.num_features_to_select = int(0.1 * X.shape[1])
        feature_scores = self.compute_scores(X, y)
        selected_features_indices = np.argsort(feature_scores)[::-1][
            : self.num_features_to_select
        ]
        return feature_scores, selected_features_indices

    def compute_scores(self, X, y):
        """
        Computes the scores for each feature. This method must be implemented by subclasses.

        Args:
            X (array-like, shape = [n_samples, n_features]): Training input samples.
            y (array-like, shape = [n_samples] or [n_samples, n_outputs]): Target values (class labels for classification, real numbers for regression).

        Returns:
            array-like: The scores for each feature.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError("Subclasses must implement compute_scores method")
