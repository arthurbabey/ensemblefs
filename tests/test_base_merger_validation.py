import pytest

from moosefs.core.feature import Feature
from moosefs.merging_strategies import L2NormMerger


def test_validate_input_rejects_non_feature_items():
    merger = L2NormMerger()
    with pytest.raises(ValueError):
        # inner items are ints, not Feature
        merger.merge([[1, 2, 3]], num_features_to_select=2)


def test_validate_input_rejects_empty_inner_lists():
    merger = L2NormMerger()
    with pytest.raises(ValueError):
        merger.merge([[]], num_features_to_select=1)

