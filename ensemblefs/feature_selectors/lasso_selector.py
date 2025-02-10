import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso

from .base_selector import FeatureSelector


class LassoSelector(FeatureSelector):
    """
    A feature selector using Lasso regression to compute feature scores.
    """

    name = "Lasso"

    def __init__(self, task, num_features_to_select=None, **kwargs):
        super().__init__(task, num_features_to_select)
        self.kwargs = kwargs

    def compute_scores(self, X, y):
        """
        Computes feature scores using Lasso regression coefficients.

        Args:
            X (DataFrame or ndarray): Training input samples.
            y (Series or ndarray): Target values.

        Returns:
            list: Scores for each feature, ordered according to X.columns.
        """
        # Ensure X is a DataFrame
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

        # Ensure y is a 1D array
        if isinstance(y, np.ndarray) and y.ndim == 2:
            y = y.ravel()

        # Set alpha with a default of 0.05, but allow override via kwargs
        alpha = self.kwargs.get("alpha", 0.05)

        # Initialize and fit Lasso
        model = Lasso(alpha=alpha, **self.kwargs)
        model.fit(X, y)

        # Get absolute coefficients as feature importance scores
        scores = np.abs(model.coef_)

        return list(scores)
