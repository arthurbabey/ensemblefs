import importlib
import sys
import warnings


def test_ensemblefs_shim_emits_deprecation():
    # Ensure a fresh import each time
    sys.modules.pop("ensemblefs", None)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mod = importlib.import_module("ensemblefs")

        # DeprecationWarning emitted
        assert any(issubclass(rec.category, DeprecationWarning) for rec in w)

        # re-export works
        assert hasattr(mod, "FeatureSelectionPipeline")

