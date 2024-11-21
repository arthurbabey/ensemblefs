import numpy as np
from sklearn.svm import SVC, SVR

from .base_selector import FeatureSelector


class SVMSelector(FeatureSelector):
    """
    A feature selector using SVM-based feature importance for feature selection.
    This class extends the FeatureSelector base class and uses SVM coefficients to evaluate features.

    Attributes:
        task (str): Specifies the machine learning task, either 'classification' or 'regression'.
        num_features_to_select (int): The number of top features to select based on importance.
            If None, a default selection logic can be applied based on a percentage of features.
        kwargs (dict): Additional keyword arguments to pass to the SVM model.
    """

    name = "SVM"

    def __init__(self, task="classification", num_features_to_select=None, **kwargs):
        """
        Initializes the SVMFeatureSelector with the specified task, number of features, and additional parameters.

        Args:
            task (str): The machine learning task ("classification" or "regression"). Defaults to "classification".
            num_features_to_select (int, optional): The number of features to select. If None, selection defaults to 10% of features.
            **kwargs: Arbitrary keyword arguments that are passed to the SVM model.
        """
        super().__init__(task, num_features_to_select)
        self.kwargs = kwargs

    def compute_scores(self, X, y):
        """
        Computes the feature importances using an SVM model.

        Args:
            X (array-like, shape = [n_samples, n_features]): Training input samples.
            y (array-like, shape = [n_samples] or [n_samples, n_outputs]): Target values (class labels for classification, real numbers for regression).

        Returns:
            array-like: The feature importances derived from the SVM model coefficients.

        Raises:
            ValueError: If the task is not 'classification' or 'regression'.
        """
        # Choose the SVM model based on the task type
        if self.task == "classification":
            model = SVC(kernel="linear", **self.kwargs)
        elif self.task == "regression":
            # fix SVR that does not accept random_state
            filtered_kwargs = {
                k: v if k != "random_state" else None
                for k, v in self.kwargs.items()
                if k != "random_state"
            }
            model = SVR(kernel="linear", **filtered_kwargs)
        else:
            raise ValueError("Task must be 'classification' or 'regression'")

        # Fit the model
        model.fit(X, y)

        # Extract feature importances from coefficients
        feature_scores = np.abs(model.coef_[0])

        return feature_scores
