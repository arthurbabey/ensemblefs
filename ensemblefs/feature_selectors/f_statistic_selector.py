from sklearn.feature_selection import f_classif, f_regression

from .base_selector import FeatureSelector


class FStatisticSelector(FeatureSelector):
    """
    A feature selector using F-statistic scores for feature selection.
    This class extends the FeatureSelector base class and uses F-statistic scores to evaluate features.

    Attributes:
        task (str): Specifies the machine learning task, either 'classification' or 'regression'.
        num_features_to_select (int): The number of top features to select based on importance.
            If None, a default selection logic can be applied based on a percentage of features.
        kwargs (dict): Additional keyword arguments to pass to the scoring function.
    """

    name = "FStatistic"

    def __init__(self, task="classification", num_features_to_select=None, **kwargs):
        """
        Initializes the FStatisticSelector with the specified task, number of features, and additional parameters.

        Args:
            task (str): The machine learning task ("classification" or "regression"). Defaults to "classification".
            num_features_to_select (int, optional): The number of features to select. If None, selection defaults to 10% of features.
            **kwargs: Arbitrary keyword arguments that are passed to the scoring function.
        """
        super().__init__(task, num_features_to_select)
        self.kwargs = kwargs

    def compute_scores(self, X, y):
        """
        Computes the F-statistic scores for feature selection.

        Args:
            X (array-like, shape = [n_samples, n_features]): Training input samples.
            y (array-like, shape = [n_samples] or [n_samples, n_outputs]): Target values (class labels for classification, real numbers for regression).

        Returns:
            array-like: The F-statistic scores for each feature.

        Raises:
            ValueError: If the task is not 'classification' or 'regression'.
        """
        if self.task == "classification":
            score_func = f_classif
        elif self.task == "regression":
            score_func = f_regression
        else:
            raise ValueError("Task must be 'classification' or 'regression'")

        # Pass additional keyword arguments to the score function
        return score_func(X, y, **self.kwargs)[0]
