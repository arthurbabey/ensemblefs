from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from .base_selector import FeatureSelector


class RandomForestSelector(FeatureSelector):
    """
    A feature selector for RandomForest models that extends the FeatureSelector base class.
    This class utilizes the built-in feature importance of RandomForest models to select the top features.

    Attributes:
        task (str): Specifies the machine learning task, either 'classification' or 'regression'.
        num_features_to_select (int): The number of top features to select based on importance.
            If None, a default selection logic can be applied based on a percentage of features.
        kwargs (dict): Additional keyword arguments to pass to the RandomForest model.
    """

    name = "RandomForest"

    def __init__(self, task, num_features_to_select=None, **kwargs):
        """
        Initializes the RandomForestFeatureSelector with the specified task, number of features, and additional parameters.

        Args:
            task (str): The machine learning task ("classification" or "regression").
            num_features_to_select (int, optional): The number of features to select. If None, selection defaults to 10% of features.
            **kwargs: Arbitrary keyword arguments that are passed to the RandomForest model.
        """
        super().__init__(task, num_features_to_select)
        self.kwargs = kwargs

    def compute_scores(self, X, y):
        """
        Computes the feature importances using a RandomForest model tailored to the specified task.

        Args:
            X (array-like, shape = [n_samples, n_features]): Training input samples.
            y (array-like, shape = [n_samples] or [n_samples, n_outputs]): Target values (class labels for classification, real numbers for regression).

        Returns:
            array-like: The feature importances derived from the RandomForest model, used as scores for feature selection.

        Raises:
            ValueError: If the task is not 'classification' or 'regression'.
        """
        if self.task == "classification":
            model = RandomForestClassifier(**self.kwargs)
        elif self.task == "regression":
            model = RandomForestRegressor(**self.kwargs)
        else:
            raise ValueError("Task must be 'classification' or 'regression'")
        model.fit(X, y)
        return model.feature_importances_
