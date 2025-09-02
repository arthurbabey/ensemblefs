from __future__ import annotations
from typing import Dict, Union
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from .base_selector import FeatureSelector


class ElasticNetSelector(FeatureSelector):
    """Elastic‑net based selector.

    • regression  → sklearn.linear_model.ElasticNet (L1+L2 on y∈ℝ)  
    • classification → sklearn.linear_model.LogisticRegression with penalty='elasticnet' (solver='saga')

    Scores are |coef| (mean over classes if multiclass).
    """

    name = "ElasticNet"

    def __init__(self, task: str, num_features_to_select: int, **kwargs: Dict) -> None:
        super().__init__(task, num_features_to_select)
        self.kwargs = kwargs

    def compute_scores(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series, pd.DataFrame],
    ) -> np.ndarray:
        # Ensure tabular objects for column-safe slicing later in the pipeline
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        if isinstance(y, (pd.DataFrame, np.ndarray)) and getattr(y, "ndim", 1) == 2:
            y = np.ravel(y)

        if self.task == "regression":
            params = {
                "alpha": self.kwargs.pop("alpha", 1.0),
                "l1_ratio": self.kwargs.pop("l1_ratio", 0.5),
                "max_iter": self.kwargs.pop("max_iter", 100_000),
                **self.kwargs,
            }
            model = make_pipeline(
                StandardScaler(with_mean=True, with_std=True),
                ElasticNet(**params)
            )
            # Fit, silencing only ConvergenceWarning (optional but useful)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                model.fit(X, y)
            #model = ElasticNet(**params)
            model.fit(X, y)
            coef = model[-1].coef_

        elif self.task == "classification":
            # LogisticRegression uses C instead of alpha; keep both if user passes.
            l1_ratio = self.kwargs.pop("l1_ratio", 0.5)
            C = self.kwargs.pop("C", 1.0)
            params = {
                "penalty": "elasticnet",
                "solver": "saga",
                "l1_ratio": l1_ratio,
                "C": C,
                "max_iter": self.kwargs.pop("max_iter", 100_000),
                **self.kwargs,
            }
            model = LogisticRegression(**params)
            model.fit(X, y)
            coef = model.coef_  # shape (n_classes, n_features) or (1, n_features)
            if coef.ndim > 1:
                coef = np.mean(np.abs(coef), axis=0)
        else:
            raise ValueError("Task must be 'classification' or 'regression'.")

        scores = np.abs(coef) if isinstance(coef, np.ndarray) else np.abs(np.asarray(coef))
        return scores
