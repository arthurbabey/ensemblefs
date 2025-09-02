import numpy as np
import pandas as pd
from .base_selector import FeatureSelector


class VarianceSelector(FeatureSelector):
    name = "Variance"

    def __init__(self, task: str, num_features_to_select: int, **kwargs):
        super().__init__(task, num_features_to_select)

    def compute_scores(self, X, y):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        return X.var(ddof=0).values  # base class will keep the highest variances
