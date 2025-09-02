import pytest

from moosefs.core.feature import Feature
from moosefs.merging_strategies import ConsensusMerger


def to_feature_grid(grid):
    # Build ragged subsets (variable lengths), as produced by the pipeline
    return [[Feature(name=str(n), score=float(n)) for n in row] for row in grid]


def test_consensus_basic_threshold():
    # Feature '2' appears twice, '3' appears three times, '1' once
    grid = to_feature_grid([[1, 2], [2, 3], [2, 3, 4]])
    merger = ConsensusMerger(k=2, fill=False)
    selected = merger.merge(grid, num_features_to_select=None)
    assert selected == {"2", "3"}


def test_consensus_fill_behaviour_trim_and_pad():
    grid = to_feature_grid([[1, 2], [2, 3], [2, 3, 4]])
    merger = ConsensusMerger(k=3, fill=True)
    # only '2' and '3' hit consensus >=3? here: '2' (3 times) and '3' (2 times)
    # with k=3 only '2' is in core; request 3 features to force padding
    out = merger.merge(grid, num_features_to_select=3)
    assert len(out) == 3
    assert "2" in out


def test_consensus_requires_num_features_when_fill():
    merger = ConsensusMerger(k=2, fill=True)
    with pytest.raises(ValueError):
        merger.merge(to_feature_grid([[1, 2], [2, 3]]))
