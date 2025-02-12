import random
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .core import Feature, ParetoAnalysis
from .metrics.stability_metrics import compute_stability_metrics
from .utils import extract_params, get_class_info


class FeatureSelectionPipeline:
    def __init__(
        self,
        data: pd.DataFrame,
        fs_methods: List[Union[str, type]],
        merging_strategy: Union[str, type],
        num_repeats: int,
        num_features_to_select: Optional[int],
        metrics: List[Union[str, type]] = ["logloss", "f1_score", "accuracy"],
        task: str = "classification",
        min_group_size: int = 2,
        fill: bool = False,
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None,
    ) -> None:
        """
        Initializes the feature selection pipeline.

        Args:
            data: DataFrame including a 'target' column.
            fs_methods: List of feature selection methods (classes or strings).
            merging_strategy: Merging strategy (class or string).
            num_repeats: Number of repeats for the pipeline.
            num_features_to_select: Desired number of features to select.
                (For rank-based merging, this must be provided.)
            metrics: List of performance metrics (classes or strings).
            task: 'classification' or 'regression'.
            min_group_size: Minimum number of methods in each subgroup.
            fill: If True, ensures exactly num_features_to_select features.
            random_state: Seed for reproducibility.
            n_jobs: Number of parallel jobs (-1 uses all processors).

        Raises:
            ValueError: If the task is invalid or if num_features_to_select is missing for rank-based merging.
        """
        self._validate_task(task)
        self.data = data
        self.task = task
        self.num_repeats = num_repeats
        self.num_features_to_select = num_features_to_select
        self.random_state = (
            random_state if random_state is not None else random.randint(0, 1000)
        )
        self.n_jobs = n_jobs

        self.fs_methods = [self._load_class(m, instantiate=True) for m in fs_methods]
        self.metrics = [self._load_class(m, instantiate=True) for m in metrics]
        self.merging_strategy = self._load_class(merging_strategy, instantiate=True)

        if self.num_features_to_select is None:
            raise ValueError("num_features_to_select must be provided")

        self.min_group_size = min_group_size
        self.fill = fill
        self.subgroup_names: List[Tuple[str, ...]] = self._generate_subgroup_names(
            self.min_group_size
        )
        self.FS_subsets: Dict[Tuple[int, str], List[Feature]] = {}
        self.merged_features: Dict[Tuple[int, Tuple[str, ...]], Any] = {}

    @staticmethod
    def _validate_task(task: str) -> None:
        if task not in ["classification", "regression"]:
            raise ValueError("Task must be either 'classification' or 'regression'.")

    def _load_class(
        self, class_or_str: Union[str, type], instantiate: bool = False
    ) -> Any:
        """
        Loads a class given a string or returns the class if already provided.

        Args:
            class_or_str: The class or its name.
            instantiate: If True, returns an instance of the class.

        Returns:
            The class or its instance.
        """
        cls = (
            get_class_info(class_or_str)
            if isinstance(class_or_str, str)
            else class_or_str
        )
        return cls() if instantiate else cls

    def _generate_subgroup_names(self, min_group_size: int) -> List[Tuple[str, ...]]:
        """
        Generates all unique combinations of feature selection method names.

        Args:
            min_group_size: Minimum size of each combination.

        Returns:
            List of tuples containing method names.
        """
        fs_method_names = [fs_method.name for fs_method in self.fs_methods]
        if min_group_size > len(fs_method_names):
            raise ValueError(
                f"Minimum group size of {min_group_size} exceeds available methods ({len(fs_method_names)})."
            )
        return [
            combo
            for r in range(min_group_size, len(fs_method_names) + 1)
            for combo in combinations(fs_method_names, r)
        ]

    def run(self, verbose: bool = True) -> Tuple[Any, int, Tuple[str, ...]]:
        """
        Runs the complete feature selection pipeline.

        Returns:
            A tuple containing:
              - The merged features from the best group,
              - The index of the best repeat,
              - The best subgroup (tuple of method names).
        """
        self._set_random_seed()

        # Initialize list of dictionaries for metrics (last dict reserved for stability)
        num_metrics = len(self.metrics) + 1
        result_dicts: List[Dict[Tuple[int, Tuple[str, ...]], float]] = [
            {} for _ in range(num_metrics)
        ]

        for i in tqdm(range(self.num_repeats), desc="Pipeline Progress"):
            train_data, test_data = self._split_data(
                test_size=0.20, random_state=self.random_state + i
            )
            self._compute_subset(train_data, i)
            self._compute_merging(i, verbose=verbose)
            result_dicts = self._compute_metrics(train_data, test_data, result_dicts, i)

        means_list = self._calculate_means(result_dicts, self.subgroup_names)
        means_list = self._replace_none(means_list)

        # Append diversity measure (number of methods in the subgroup)
        metrics_with_diversity = [
            mean + [len(group)] for mean, group in zip(means_list, self.subgroup_names)
        ]
        best_group = self._compute_pareto(metrics_with_diversity, self.subgroup_names)
        best_group_metrics = self._extract_repeat_metrics(best_group, *result_dicts)
        best_group_metrics = self._replace_none(best_group_metrics)
        best_repeat = self._compute_pareto(
            best_group_metrics, [str(i) for i in range(self.num_repeats)]
        )
        return (
            self.merged_features[(int(best_repeat), best_group)],
            int(best_repeat),
            best_group,
        )

    def _replace_none(self, metrics: List[List[Optional[float]]]) -> List[List[float]]:
        """
        Replaces groups containing None with lists of -inf.

        Args:
            metrics: List of metric lists per group.

        Returns:
            Cleaned list with no None values.
        """
        return [
            (
                group_metrics
                if all(metric is not None for metric in group_metrics)
                else [-float("inf")] * len(group_metrics)
            )
            for group_metrics in metrics
        ]

    def _set_random_seed(self) -> None:
        seed = int(self.random_state)
        np.random.seed(seed)
        random.seed(seed)

    def _split_data(
        self, test_size: float, random_state: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        stratify = self.data["target"] if self.task == "classification" else None
        return train_test_split(
            self.data, test_size=test_size, random_state=random_state, stratify=stratify
        )

    def _compute_subset(self, train_data: pd.DataFrame, idx: int) -> None:
        """
        Computes feature subsets using each feature selection method.

        Args:
            train_data: Training dataset.
            idx: Current repeat index.
        """
        X_train = train_data.drop("target", axis=1)
        y_train = train_data["target"]
        feature_names = X_train.columns.tolist()

        for fs_method in self.fs_methods:
            method_name = fs_method.name
            scores, indices = fs_method.select_features(X_train, y_train)
            self.FS_subsets[(idx, method_name)] = [
                Feature(
                    name,
                    score=scores[i] if scores is not None else None,
                    selected=(i in indices),
                )
                for i, name in enumerate(feature_names)
            ]

    def _compute_merging(self, idx: int, verbose: bool = False) -> None:
        """
        Merges feature subsets for each subgroup.

        Args:
            idx: Current repeat index.
            verbose: If True, prints warnings when no merged features are produced.
        """
        for group in self.subgroup_names:
            merged = self._merge_group_features(idx, group)
            if merged:
                self.merged_features[(idx, group)] = merged
            elif verbose:
                print(f"Warning: {group} produced no merged features.")

    def _merge_group_features(self, idx: int, group: Tuple[str, ...]) -> Any:
        """
        Merges features for a specific group of methods.

        Args:
            idx: Current repeat index.
            group: Tuple of method names.

        Returns:
            Merged features as produced by the merging strategy.
        """
        group_features = [
            [feature for feature in self.FS_subsets[(idx, method)] if feature.selected]
            for method in group
        ]
        return self.merging_strategy.merge(
            group_features, self.num_features_to_select, fill=self.fill
        )

    def _compute_performance_metrics(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> List[float]:
        """
        Computes performance metrics using all metric methods.

        Returns:
            List of averaged metric values.
        """
        return [
            metric.compute(X_train, y_train, X_test, y_test) for metric in self.metrics
        ]

    def _compute_metrics(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        result_dicts: List[Dict[Tuple[int, Tuple[str, ...]], float]],
        idx: int,
    ) -> List[Dict[Tuple[int, Tuple[str, ...]], float]]:
        """
        Computes and stores performance and stability metrics for each subgroup.

        Args:
            train_data: Training dataset.
            test_data: Test dataset.
            result_dicts: List of dictionaries to store metrics.
            idx: Current repeat index.

        Returns:
            Updated result_dicts with computed metrics.
        """
        for group in self.subgroup_names:
            key = (idx, group)
            if key not in self.merged_features:
                continue

            features = list(self.merged_features[key])
            X_train_subset, y_train = train_data[features], train_data["target"]
            X_test_subset, y_test = test_data[features], test_data["target"]

            metrics_vals = self._compute_performance_metrics(
                X_train_subset, y_train, X_test_subset, y_test
            )
            for i, val in enumerate(metrics_vals):
                result_dicts[i][key] = val

            features_stability = [
                [
                    feature.name
                    for feature in self.FS_subsets[(idx, method)]
                    if feature.selected
                ]
                for method in group
            ]
            stability = (
                compute_stability_metrics(features_stability)
                if features_stability
                else 0
            )
            result_dicts[len(metrics_vals)][key] = stability

        return result_dicts

    @staticmethod
    def _calculate_means(
        result_dicts: List[Dict[Tuple[int, Tuple[str, ...]], float]],
        group_names: List[Tuple[str, ...]],
    ) -> List[List[Optional[float]]]:
        """
        Calculates the mean metrics for each subgroup.

        Args:
            result_dicts: List of dictionaries with metrics.
            group_names: List of subgroup names.

        Returns:
            List of lists containing mean metrics per group.
        """
        means_list = []
        for group in group_names:
            group_means = [
                (
                    None
                    if np.isnan(
                        np.mean(
                            [value for (idx, name), value in d.items() if name == group]
                        )
                    )
                    else np.mean(
                        [value for (idx, name), value in d.items() if name == group]
                    )
                )
                for d in result_dicts
            ]
            means_list.append(group_means)
        return means_list

    @staticmethod
    def _compute_pareto(groups: List[List[float]], names: List[Any]) -> Any:
        """
        Performs Pareto analysis to identify the best-performing group or repeat.

        Args:
            groups: List of metric lists.
            names: List of corresponding names.

        Returns:
            The name of the best-performing group or repeat.
        """
        pareto = ParetoAnalysis(groups, names)
        pareto_results = pareto.get_results()
        return pareto_results[0][0]

    @staticmethod
    def _extract_repeat_metrics(
        group: Union[str, Tuple[str, ...]],
        *result_dicts: Dict[Tuple[int, Tuple[str, ...]], float],
    ) -> List[List[Optional[float]]]:
        """
        Extracts metrics across repeats for a given group.

        Args:
            group: The subgroup name.
            result_dicts: Dictionaries with metrics per repeat.

        Returns:
            A 2D list of metrics with each row corresponding to a repeat.
        """
        indices = sorted({key[0] for key in result_dicts[0].keys()})
        result_array = []
        for idx in indices:
            row = [d.get((idx, group)) for d in result_dicts]
            result_array.append(row)
        return result_array

    def _load_class(self, input: Union[str, object], instantiate: bool = False) -> Any:
        """
        Loads a class or returns an instance if already provided.

        Args:
            input: A string identifier or an instance of a feature selector/merging strategy.
            instantiate: If True, instantiates the class with extracted parameters.

        Returns:
            The class or its instance.

        Raises:
            ValueError: If input is not a valid identifier or instance.
        """
        if isinstance(input, str):
            cls, params = get_class_info(input)
            if instantiate:
                init_params = extract_params(cls, self, params)
                return cls(**init_params)
            return cls
        elif hasattr(input, "select_features") or hasattr(input, "merge"):
            # Assumes valid instance if it has a 'select_features' or 'merge' method.
            return input
        else:
            raise ValueError(
                "Input must be a string identifier or a valid instance of a feature selector or merging strategy."
            )

    def __str__(self) -> str:
        return (
            f"Feature selection pipeline with: merging strategy: {self.merging_strategy}, "
            f"feature selection methods: {self.fs_methods}, "
            f"number of repeats: {self.num_repeats}"
        )

    def __call__(self) -> Any:
        return self.run()
