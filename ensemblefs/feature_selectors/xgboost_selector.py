from xgboost import XGBClassifier, XGBRegressor

from .base_selector import FeatureSelector


class XGBoostSelector(FeatureSelector):
    """
    A feature selector for XGBoost models that extends the FeatureSelector class.
    This class is tailored to utilize XGBoost's built-in feature importance to select features.

    Attributes:
        task (str): The type of machine learning task ("classification" or "regression").
        num_features_to_select (int): The number of top features to select based on importance.
            If None, a default selection logic can be applied based on a percentage of features.
        kwargs (dict): Additional keyword arguments to pass to the XGBoost model constructor.
    """

    name = "XGBoost"

    def __init__(self, task="classification", num_features_to_select=None, **kwargs):
        """
        Initializes the XGBoostFeatureSelector with the specified task, number of features, and any additional parameters.

        Args:
            task (str): The machine learning task ("classification" or "regression"). Defaults to "classification".
            num_features_to_select (int, optional): The number of features to select. If None, selection defaults to 10% of features.
            **kwargs: Arbitrary keyword arguments that are passed to the XGBoost model.
        """
        super().__init__(task, num_features_to_select)
        self.kwargs = kwargs

    def compute_scores(self, X, y):
        """
        Computes the feature importances using an XGBoost model.

        Args:
            X (array-like, shape = [n_samples, n_features]): The training input samples.
            y (array-like, shape = [n_samples] or [n_samples, n_outputs]): The target values (class labels for classification, real numbers for regression).

        Returns:
            array-like: The feature importances derived from the XGBoost model.
        """
        if self.task == "classification":
            model = XGBClassifier(**self.kwargs)
        elif self.task == "regression":
            model = XGBRegressor(**self.kwargs)
        model.fit(X, y)
        return model.feature_importances_
