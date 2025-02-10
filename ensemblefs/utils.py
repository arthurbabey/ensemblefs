import inspect

# Mapping of class identifiers to their import paths and initialization parameters.
# Template:
#     "identifier": ("module.path.ClassName", ["param1", "param2", ...])
class_path_mapping = {
    "mae": (
        "ensemblefs.metrics.performance_metrics.MeanSquaredError",
        [],
    ),
    "mse": (
        "ensemblefs.metrics.performance_metrics.MeanAbsoluteError",
        [],
    ),
    "r2_score": (
        "ensemblefs.metrics.performance_metrics.R2Score",
        [],
    ),
    "logloss": (
        "ensemblefs.metrics.performance_metrics.LogLoss",
        [],
    ),
    "f1_score": (
        "ensemblefs.metrics.performance_metrics.F1Score",
        [],
    ),
    "accuracy": (
        "ensemblefs.metrics.performance_metrics.Accuracy",
        [],
    ),
    "precision_score": (
        "ensemblefs.metrics.performance_metrics.PrecisionScore",
        [],
    ),
    "recall_score": (
        "ensemblefs.metrics.performance_metrics.RecallScore",
        [],
    ),
    "f_statistic_selector": (
        "ensemblefs.feature_selectors.f_statistic_selector.FStatisticSelector",
        ["task", "num_features_to_select"],
    ),
    "random_forest_selector": (
        "ensemblefs.feature_selectors.random_forest_selector.RandomForestSelector",
        ["task", "num_features_to_select", "random_state", "n_jobs"],
    ),
    "mutual_info_selector": (
        "ensemblefs.feature_selectors.mutual_info_selector.MutualInfoSelector",
        ["task", "num_features_to_select", "n_jobs"],
    ),
    "rfe_rf_selector": (
        "ensemblefs.feature_selectors.rfe_rf_selector.RFERFSelector",
        ["task", "num_features_to_select", "n_jobs"],
    ),
    "svm_selector": (
        "ensemblefs.feature_selectors.svm_selector.SVMSelector",
        ["task", "num_features_to_select", "random_state"],
    ),
    "xgboost_selector": (
        "ensemblefs.feature_selectors.xgboost_selector.XGBoostSelector",
        ["task", "num_features_to_select", "random_state", "n_jobs"],
    ),
    "mrmr_selector": (
        "ensemblefs.feature_selectors.mrmr_selector.MRMRSelector",
        ["task", "num_features_to_select", "n_jobs"],
    ),
    "lasso_selector": (
        "ensemblefs.feature_selectors.lasso_selector.LassoSelector",
        ["task", "num_features_to_select", "random_state"],
    ),
    "borda_merger": (
        "ensemblefs.merging_strategies.borda_merger.BordaMerger",
        [],
    ),
    "kemeny_young_merger": (
        "ensemblefs.merging_strategies.kemeny_young_merger.KemenyYoungMerger",
        [],
    ),
    "union_of_intersections_merger": (
        "ensemblefs.merging_strategies.union_of_intersections_merger.UnionOfIntersectionsMerger",
        [],
    ),
}


def dynamic_import(class_path):
    """
    Dynamically imports a class based on its full class path.

    Args:
        class_path (str): The full dot-separated path to the class, e.g., 'ensemblefs.module.ClassName'.

    Returns:
        type: The class object specified by class_path.
    """
    components = class_path.split(".")
    module_path = ".".join(components[:-1])
    class_name = components[-1]
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)


def get_class_info(identifier):
    """
    Retrieves the class object and its expected parameters based on a string identifier.

    Args:
        identifier (str): A string representing a class identifier in `class_path_mapping`.

    Returns:
        tuple: (class object, list of expected parameters)

    Raises:
        ValueError: If the identifier is not found in `class_path_mapping`.
    """
    if identifier not in class_path_mapping:
        raise ValueError(f"Unknown class identifier: {identifier}")
    class_path, params = class_path_mapping[identifier]
    cls = dynamic_import(class_path)
    return cls, params


def extract_params(cls, instance, params):
    """
    Extracts and returns the initialization parameters required by the class,
    based on the class signature and the provided instance attributes.

    Args:
        cls (type): The class object to be instantiated.
        instance (object): The instance from which to pull attribute values for initialization.
        params (list): A list of parameter names expected by the class.

    Returns:
        dict: A dictionary of parameters (name-value pairs) for instantiating the class.
    """
    sig = inspect.signature(cls.__init__)

    # Extract explicitly defined parameters
    extracted_params = {
        param: getattr(instance, param)
        for param in params
        if param in sig.parameters and hasattr(instance, param)
    }

    # If **kwargs exists in the class signature, capture additional params
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        additional_params = {
            param: getattr(instance, param)
            for param in params
            if param not in sig.parameters and hasattr(instance, param)
        }
        extracted_params.update(additional_params)

    return extracted_params
