import numpy as np

from moosefs.feature_selectors.default_variance import variance_selector_default


def test_variance_selector_default_numpy_array():
    # Construct columns with variances [0, ~2.667, 24.0]; median ≈ 2.667
    # Using threshold = alpha * median, with alpha=0.5 → ~1.333
    X = np.array(
        [
            [0.0, 0.0, 0.0],  # col0 constant, col1/2 increase linearly
            [0.0, 2.0, 6.0],
            [0.0, 4.0, 12.0],
        ],
        dtype=float,
    )
    scores, idx = variance_selector_default(X, alpha=0.5)
    assert isinstance(scores, list)
    # columns with variance >= 1.333... are col1 and col2
    assert idx == [1, 2]
