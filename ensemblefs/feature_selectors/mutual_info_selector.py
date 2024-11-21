from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from .base_selector import FeatureSelector


class MutualInfoSelector(FeatureSelector):
    """
    A feature selector using mutual information scores for feature selection.
    This class extends the FeatureSelector base class and uses mutual information to evaluate features.

    Attributes:
        task (str): Specifies the machine learning task, either 'classification' or 'regression'.
        num_features_to_select (int): The number of top features to select based on importance.
            If None, a default selection logic can be applied based on a percentage of features.
        kwargs (dict): Additional keyword arguments to pass to the mutual information function.
    """

    name = "MutualInfo"

    def __init__(self, task="classification", num_features_to_select=None, **kwargs):
        """
        Initializes the MutualInfoSelector with the specified task, number of features, and additional parameters.

        Args:
            task (str): The machine learning task ("classification" or "regression"). Defaults to "classification".
            num_features_to_select (int, optional): The number of features to select. If None, selection defaults to 10% of features.
            **kwargs: Arbitrary keyword arguments that are passed to the mutual information function.
        """
        super().__init__(task, num_features_to_select)
        self.kwargs = kwargs

    def compute_scores(self, X, y):
        """
        Computes the mutual information scores for feature selection.

        Args:
            X (array-like, shape = [n_samples, n_features]): Training input samples.
            y (array-like, shape = [n_samples] or [n_samples, n_outputs]): Target values (class labels for classification, real numbers for regression).

        Returns:
            array-like: The mutual information scores for each feature.

        Raises:
            ValueError: If the task is not 'classification' or 'regression'.
        """
        if self.task == "classification":
            mutual_info_func = mutual_info_classif
        elif self.task == "regression":
            mutual_info_func = mutual_info_regression
        else:
            raise ValueError("Task must be 'classification' or 'regression'")

        # Compute mutual information scores, passing additional keyword arguments
        return mutual_info_func(X, y, **self.kwargs)
