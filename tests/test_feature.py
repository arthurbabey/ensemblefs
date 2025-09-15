import pytest

from moosefs.core.feature import Feature


@pytest.fixture
def feature_instance():
    return Feature("test_feature")


def test_feature_initialization(feature_instance):
    assert feature_instance.name == "test_feature"
    assert feature_instance.score is None
    assert not feature_instance.selected


def test_feature_setters(feature_instance):
    feature_instance.set_score(0.5)
    feature_instance.set_selected(True)

    assert feature_instance.score == 0.5
    assert feature_instance.selected


def test_feature_str_representation(feature_instance):
    expected_str = "Feature(name=test_feature, score=None, selected=False)"
    assert str(feature_instance) == expected_str


def test_feature_repr_representation(feature_instance):
    expected_repr = "Feature('test_feature', None, False)"
    assert repr(feature_instance) == expected_repr
