import numpy as np
import pandas as pd


def variance_selector_default(X, y=None, alpha=0.01):
    # ensure DataFrame for variance call
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

    # 1-D array of float64
    variances = X.var(ddof=0).to_numpy(dtype=float)

    # convert to plain Python floats to avoid dtype-object surprises
    scores = [float(v) for v in variances]

    threshold = alpha * float(np.median(variances))

    # plain Python ints for indices
    indices = [int(i) for i in np.where(variances >= threshold)[0]]

    return scores, indices
