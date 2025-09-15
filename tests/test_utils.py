import types

import pytest

from moosefs.utils import dynamic_import, extract_params, get_class_info


def test_get_class_info_known_selector():
    cls, params = get_class_info("f_statistic_selector")
    # module path and param list are defined in mapping
    assert cls.__name__ == "FStatisticSelector"
    assert set(params) >= {"task", "num_features_to_select"}


def test_get_class_info_unknown_identifier():
    with pytest.raises(ValueError):
        get_class_info("totally_unknown_component")


def test_dynamic_import_roundtrip():
    # import a known class via fully qualified path
    klass = dynamic_import("moosefs.merging_strategies.borda_merger.BordaMerger")
    assert klass.__name__ == "BordaMerger"


def test_extract_params_from_instance_like_object():
    # Build a tiny instance-like object carrying the attributes expected
    inst = types.SimpleNamespace(task="classification", num_features_to_select=7)
    cls, params = get_class_info("f_statistic_selector")
    extracted = extract_params(cls, inst, params)
    assert extracted == {
        "task": "classification",
        "num_features_to_select": 7,
    }
