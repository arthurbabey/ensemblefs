import inspect
from typing import Any, Dict, List, Tuple

# Mapping of class identifiers to their import paths and expected initialization parameters.
# Template: "identifier": ("module.path.ClassName", ["param1", "param2", ...])
class_path_mapping: Dict[str, Tuple[str, List[str]]] = {
    "mae": (
        "moosefs.metrics.performance_metrics.MeanSquaredError",
        [],
    ),
    "mse": (
        "moosefs.metrics.performance_metrics.MeanAbsoluteError",
        [],
    ),
    "r2_score": (
        "moosefs.metrics.performance_metrics.R2Score",
        [],
    ),
    "logloss": (
        "moosefs.metrics.performance_metrics.LogLoss",
        [],
    ),
    "f1_score": (
        "moosefs.metrics.performance_metrics.F1Score",
        [],
    ),
    "accuracy": (
        "moosefs.metrics.performance_metrics.Accuracy",
        [],
    ),
    "precision_score": (
        "moosefs.metrics.performance_metrics.PrecisionScore",
        [],
    ),
    "recall_score": (
        "moosefs.metrics.performance_metrics.RecallScore",
        [],
    ),
    "f_statistic_selector": (
        "moosefs.feature_selectors.f_statistic_selector.FStatisticSelector",
        ["task", "num_features_to_select"],
    ),
    "random_forest_selector": (
        "moosefs.feature_selectors.random_forest_selector.RandomForestSelector",
        ["task", "num_features_to_select", "random_state"],
    ),
    "mutual_info_selector": (
        "moosefs.feature_selectors.mutual_info_selector.MutualInfoSelector",
        ["task", "num_features_to_select", "random_state"],
    ),
    "svm_selector": (
        "moosefs.feature_selectors.svm_selector.SVMSelector",
        ["task", "num_features_to_select"],
    ),
    "xgboost_selector": (
        "moosefs.feature_selectors.xgboost_selector.XGBoostSelector",
        ["task", "num_features_to_select", "random_state"],
    ),
    "mrmr_selector": (
        "moosefs.feature_selectors.mrmr_selector.MRMRSelector",
        ["task", "num_features_to_select"],
    ),
    "lasso_selector": (
        "moosefs.feature_selectors.lasso_selector.LassoSelector",
        ["task", "num_features_to_select", "random_state"],
    ),
    "elastic_net_selector": (
        "moosefs.feature_selectors.elastic_net_selector.ElasticNetSelector",
        ["task", "num_features_to_select", "random_state"],
    ),
    "variance_selector": (
        "moosefs.feature_selectors.variance_selectors.VarianceSelector",
        ["task", "num_features_to_select"],
    ),
    "borda_merger": (
        "moosefs.merging_strategies.borda_merger.BordaMerger",
        [],
    ),
    "union_of_intersections_merger": (
        "moosefs.merging_strategies.union_of_intersections_merger.UnionOfIntersectionsMerger",
        [],
    ),
    "borda_merger": (
        "moosefs.merging_strategies.borda_merger.BordaMerger",
        [],
    ),
    "l2_norm_merger": (
        "moosefs.merging_strategies.l2_norm_merger.L2NormMerger",
        [],
    ),
    "arithmetic_mean_merger": (
        "moosefs.merging_strategies.arithmetic_mean_merger.ArithmeticMeanMerger",
        [],
    ),
    "consensus_merger": (
        "moosefs.merging_strategies.consensus_merger.ConsensusMerger",
        ["k", "fill"],
    ),
}


def dynamic_import(class_path: str) -> type:
    """
    Dynamically imports a class based on its full class path.

    Args:
        class_path: Full dot-separated path to the class (e.g., 'moosefs.module.ClassName').

    Returns:
        The class object specified by class_path.
    """
    components = class_path.split(".")
    module_path = ".".join(components[:-1])
    class_name = components[-1]
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)


def get_class_info(identifier: str) -> Tuple[type, List[str]]:
    """
    Retrieves the class object and its expected parameters based on a string identifier.

    Args:
        identifier: A string representing a class identifier defined in `class_path_mapping`.

    Returns:
        A tuple containing the class object and a list of expected parameter names.

    Raises:
        ValueError: If the identifier is not found in `class_path_mapping`.
    """
    if identifier not in class_path_mapping:
        raise ValueError(f"Unknown class identifier: {identifier}")
    class_path, params = class_path_mapping[identifier]
    cls = dynamic_import(class_path)
    return cls, params


def extract_params(cls: type, instance: Any, params: List[str]) -> Dict[str, Any]:
    """
    Extracts and returns the initialization parameters required by the class,
    based on the class signature and attributes of the provided instance.

    Args:
        cls: The class object to be instantiated.
        instance: The instance from which attribute values are extracted.
        params: A list of parameter names expected by the class.

    Returns:
        A dictionary of parameters (name-value pairs) for instantiating the class.
    """
    sig = inspect.signature(cls.__init__)

    extracted_params: Dict[str, Any] = {
        param: getattr(instance, param)
        for param in params
        if param in sig.parameters and hasattr(instance, param)
    }

    # If **kwargs exists in the class signature, include additional parameters.
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        additional_params = {
            param: getattr(instance, param)
            for param in params
            if param not in sig.parameters and hasattr(instance, param)
        }
        extracted_params.update(additional_params)

    return extracted_params
