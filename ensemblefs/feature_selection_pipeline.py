import random
from itertools import combinations

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .core import Feature, ParetoAnalysis
from .metrics.stability_metrics import compute_stability_metrics
from .utils import extract_params, get_class_info


class FeatureSelectionPipeline:
    def __init__(
        self,
        data,
        fs_methods,
        merging_strategy,
        num_repeats,
        num_features_to_select,
        metrics=["logloss", "f1_score", "accuracy"],
        task="classification",
        min_group_size=2,
        random_state=None,
        n_jobs=None,
    ):
        """
        Initializes a FeatureSelectionPipeline object, which configures and executes
        a feature selection process using specified methods and strategies.

        Args:
            data (pandas.DataFrame): The dataset to process, which must include a 'target' column.
            fs_methods (list of classes or str): Feature selection methods to be used. Each entry
                can be either a class or a string that corresponds to a defined class name. These
                classes should implement a method for feature selection.
            merging_strategy (class or str): The strategy used to merge features selected by
                different methods. Can be either a class or a string corresponding to a defined class name.
            num_repeats (int): The number of times the feature selection process will be repeated.
            threshold (int, optional): The number of features to select. Defaults to one-tenth of the
                number of features in 'data' if not provided.
            metrics (list of classes or str): Performance metrics to be used for evaluation. Each entry
                can be either a class or a string that corresponds to a defined class name. These classes
                should implement a method for computing the metric.
            task (str, optional): The type of task ('classification' or 'regression'). Defaults to
                'classification'.
            random_state (int, optional): Random seed for reproducibility. Defaults to None.
            n_jobs (int, optional): The number of jobs to compute the feature selection pipeline, -1 means using all processors.

        Raises:
            ValueError: If 'task' is neither 'classification' nor 'regression'.

        Example:
            >>> pipeline = FeatureSelectionPipeline(
                data=df,
                fs_methods=['RandomForestSelector', 'MutualInfoSelector'],
                merging_strategy='BordaMerger',
                num_repeats=10,
                threshold=5,
                metrics=['loglloss', 'f1_score', 'accuracy'],
                task='classification',
                random_state=42
                n_jobs=n_jobs
            )
        """
        self._validate_task(task)
        self.data = data
        self.num_repeats = num_repeats
        self.num_features_to_select = (
            num_features_to_select
            if num_features_to_select is not None
            else data.shape[1] // 10
        )
        self.task = task
        self.random_state = (
            random_state if random_state is not None else random.randint(0, 1000)
        )
        self.n_jobs = n_jobs
        self.fs_methods = [self._load_class(m, instantiate=True) for m in fs_methods]
        self.metrics = [self._load_class(m, instantiate=True) for m in metrics]
        self.merging_strategy = self._load_class(merging_strategy, instantiate=True)
        self.min_group_size = min_group_size
        self.subgroup_names = self._generate_subgroup_names(
            min_group_size=self.min_group_size
        )
        self.FS_subsets = {}
        self.merged_features = {}

    @staticmethod
    def _validate_task(task):
        if task not in ["classification", "regression"]:
            raise ValueError("Task must be either 'classification' or 'regression'.")

    def _generate_subgroup_names(self, min_group_size=2):
        """
        Generates all unique combinations of feature selection method names with sizes
        ranging from min_group_size to the total number of methods.
        Generates combinations using: nCr = n! / (r! * (n - r)!)

        Args:
            min_group_size (int): Minimum size of each combination.

        Returns:
            list: Combinations of method names.
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

    def run(self, verbose=True):
        """
        Runs the entire feature selection and classification pipeline.

        This method iterates through the pipeline for the specified number of repeats, performing:
        - Data splitting
        - Feature computation and merging
        - Performance evaluation
        - Calculation of mean metrics per repeat
        - Pareto analysis to identify best performing groups and repeats

        Returns:
        - tuple: A tuple containing the merged features, the best repeat, and the best group name.
        """

        self._set_random_seed()
        result_dicts = [
            {} for _ in range(4)
        ]  # store 3 performance metrics and 1 stability

        for i in tqdm(range(self.num_repeats), desc="Pipeline Progress"):
            train_data, test_data = self._split_data(
                test_size=0.20, random_state=self.random_state + i
            )
            self._compute_subset(train_data, i)
            self._compute_merging(i, verbose=verbose)
            result_dicts = self._compute_metrics(train_data, test_data, result_dicts, i)

        # this is list of groups where each groups are a list of mean metrics respecting the orders of result_dicts
        # first list of the list is : mean accs for group1, means AUROC for group1 etc
        list_of_means = self._calculate_means(result_dicts, self.subgroup_names)

        # replace None with -inf coming from group with no merged features from all repeats
        list_of_means = self._replace_none(list_of_means)

        # add number of methods to maximize; ensuring diversity
        metrics = [
            mean + [len(group_name)]
            for mean, group_name in zip(list_of_means, self.subgroup_names)
        ]
        metrics_names = self.subgroup_names

        # find the best group using average metrics
        best_group_name = self._compute_pareto(groups=metrics, names=metrics_names)
        best_group_metrics = self._extract_repeat_metrics(
            best_group_name, *result_dicts
        )

        # replace None with -inf coming if best_group has one empty repeat
        best_group_metrics = self._replace_none(best_group_metrics)

        # find the best repeat using metrics from best group only
        best_repeat = self._compute_pareto(
            groups=best_group_metrics,
            names=[str(i) for i in range(self.num_repeats)],
        )

        return (
            self.merged_features[(int(best_repeat), best_group_name)],
            best_repeat,
            best_group_name,
        )

    def _replace_none(self, metrics):
        metrics = [
            (
                group_metrics
                if all(metric is not None for metric in group_metrics)
                else [-float("inf")] * len(group_metrics)
            )
            for group_metrics in metrics
        ]
        return metrics

    def _set_random_seed(self):
        if self.random_state is not None:
            import random

            random_state = int(self.random_state)
            np.random.seed(random_state)
            random.seed(random_state)

    def _split_data(self, test_size, random_state):
        stratify = self.data["target"] if self.task == "classification" else None
        return train_test_split(
            self.data, test_size=test_size, random_state=random_state, stratify=stratify
        )

    def _compute_subset(self, train_data, idx):
        """
        Computes feature subset for each feature selection method per repeat.

        Parameters:
        - train_data (pandas.DataFrame): The training dataset.
        - idx (int): Index for the current iteration.
        """
        X_train, y_train = train_data.drop("target", axis=1), train_data["target"]
        feature_names = X_train.columns.tolist()

        for fs_method in self.fs_methods:
            method_name = fs_method.name
            scores, indices = fs_method.select_features(X_train, y_train)

            # add all features in FS_subsets with name, score and selected status
            self.FS_subsets[(idx, method_name)] = [
                Feature(
                    name,
                    score=scores[i] if scores is not None else None,
                    selected=(i in indices),
                )
                for i, name in enumerate(feature_names)
            ]

    def _compute_merging(self, idx, verbose=False):
        for group in self.subgroup_names:
            merged_features = self._merge_group_features(idx, group)
            if merged_features:
                self.merged_features[(idx, group)] = merged_features
            elif verbose:
                print(f"Warning: '{group}' produced no merged features.")

    def _merge_group_features(self, idx, group):
        group_features = [
            [
                feature
                for feature in self.FS_subsets[(idx, method)]
                if feature.get_selected()
            ]
            for method in group
        ]
        return self.merging_strategy.merge(group_features, self.num_features_to_select)

    def _compute_performance_metrics(self, X_train, y_train, X_test, y_test):

        metric_values = []
        for metric in self.metrics:
            m = metric.compute(X_train, y_train, X_test, y_test)
            metric_values.append(m)
        return metric_values  # list of three averaged metrics, e.g : loglloss, f1_score, accuracy

    def _compute_metrics(self, train_data, test_data, result_dicts, idx):
        for group_name in self.subgroup_names:
            # if the group has no merged features, skip
            # result_dicts wont have the key for this idx, group_name keypair
            if (idx, group_name) not in self.merged_features.keys():
                continue

            features = list(self.merged_features[(idx, group_name)])

            X_train, y_train = train_data[features], train_data["target"]
            X_test, y_test = test_data[features], test_data["target"]
            results = self._compute_performance_metrics(
                X_train, y_train, X_test, y_test
            )

            for i, result in enumerate(results):
                result_dicts[i][(idx, group_name)] = result

            features_stability = [
                [
                    feature.get_name()
                    for feature in self.FS_subsets[(idx, method_name)]
                    if feature.get_selected()
                ]
                for method_name in group_name
            ]
            stability = (
                compute_stability_metrics(features_stability)
                if features_stability
                else 0
            )
            result_dicts[3][(idx, group_name)] = stability

        return result_dicts

    @staticmethod
    def _calculate_means(result_dicts, group_names):
        """
        Calculate the mean of the metrics for each group_name,
        returning None for all metrics for group with no merged features i.e when  np.mean return np.nan
        """
        means_list = []
        for group_name in group_names:
            group_means = [
                (
                    None
                    if np.isnan(
                        np.mean(
                            [
                                value
                                for (idx, name), value in d.items()
                                if name == group_name
                            ]
                        )
                    )
                    else np.mean(
                        [
                            value
                            for (idx, name), value in d.items()
                            if name == group_name
                        ]
                    )
                )
                for d in result_dicts
            ]
            means_list.append(group_means)
        return means_list

    @staticmethod
    def _compute_pareto(groups, names):
        """
        Performs Pareto analysis to identify best-performing groups or repeats.

        Parameters:
        - groups (list of list): Group of data to perform the pareto analysis on.
        - names (list): Names of the scores for analysis.

        Returns:
        - str: Name of the best-performing group or repeat.

        Details:
        - Utilizes Pareto analysis to determine the best performing group or repeat.
        """
        pareto = ParetoAnalysis(groups, names)
        pareto_results = pareto.get_results()
        best_group_name = pareto_results[0][0]
        return best_group_name

    @staticmethod
    def _extract_repeat_metrics(group_name, *result_dicts):
        # Find all unique indices
        indices = sorted(set(key[0] for key in result_dicts[0].keys()))
        # Create a 2D array to store the metrics for the fixed group name
        result_array = []
        for idx in indices:
            row = [d.get((idx, group_name)) for d in result_dicts]
            result_array.append(row)

        return result_array

    def _load_class(self, input, instantiate=False):
        if isinstance(input, str):
            cls, params = get_class_info(input)
        elif isinstance(input, object):
            # here i should add a check to ensure object is a feature selectors or a merging strategy
            return input
        else:
            raise ValueError(
                "Input must be a string identifier or an instance of a class."
            )

        if instantiate:
            init_params = extract_params(cls, self, params)
            return cls(**init_params)
        return cls

    def __str__(self):
        return (
            f"Feature selection pipeline with : "
            f"merging strategy : {self.merging_strategy}, "
            f"feature selection methods: {self.fs_methods}, "
            f"Number of repeats: {self.num_repeats}"
        )

    def __call__(self):
        return self.run()
