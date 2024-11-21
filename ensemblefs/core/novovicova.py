import numpy as np


class StabilityNovovicova:
    """
    A class to compute the stability of feature selection algorithms based on a measure
    proposed by Novovicová, Somol, and Pudil (2009). This stability measure quantifies
    the consistency of selected features across different subsets or resamplings of a dataset.

    Attributes:
        feature_selections (list of set): A list where each set contains the features selected
                                          in one dataset.
        N (int): Total number of feature occurrences across all datasets.
        n (int): Number of datasets or subsets.

    References:
        Novovicová, J., Somol, P., & Pudil, P. (2009). “A New Measure of Feature Selection
        Algorithms' Stability.” In 2009 IEEE International Conference on Data Mining Workshops.
        doi:10.1109/icdmw.2009.32.

    Example:
        >>> feature_selections = [
                {1, 2, 3},
                {2, 3, 4, 5},
                {1, 2, 6}
            ]
        >>> stability_calculator = StabilityNovovicova(feature_selections)
        >>> print(stability_calculator.compute_stability())
        # Outputs the computed stability measure SH(S).

    """

    def __init__(self, feature_selections):
        """
        Initializes the StabilityNovovicova class with a list of lists, each representing selected features in a dataset.
        Checks and converts the lists to sets to ensure uniqueness within each dataset for simplicity in calculations.

        Args:
            feature_selections (list of lists or sets): Each sublist or set represents features selected in one dataset.
        """
        if not self.validate_inputs(feature_selections):
            raise ValueError(
                "Input validation failed, check the data format and content."
            )
        self.feature_selections = [set(sel) for sel in feature_selections]
        self.N = sum(
            len(sel) for sel in self.feature_selections
        )  # Total number of feature occurrences
        self.n = len(self.feature_selections)  # Number of datasets

    def validate_inputs(self, feature_selections):
        """
        Validates the input feature selections for non-empty data, uniformity of data types, and consistent feature representation.

        Args:
            feature_selections (list of lists or sets): Feature selections to validate.

        Returns:
            bool: True if all validations pass, False otherwise.
        """
        if not feature_selections:
            print("Feature selections cannot be empty.")
            return False

        element_type = None
        for sublist in feature_selections:
            for item in sublist:
                if element_type is None:
                    element_type = type(item)
                elif type(item) != element_type:
                    print("All elements must be of the same type.")
                    return False

        return True

    def compute_stability(self):
        """
        Computes the stability SH(S) using the provided feature selections.

        Returns:
            float: The computed stability measure SH(S), ranging from 0 (no stability) to 1 (full stability).

        Raises:
            ValueError: If N or n are 0, which prevents calculation.
        """
        if self.N == 0 or self.n == 0:
            return (
                0  # Return stability of 0 if there are no feature selections or subsets
            )

        # Check if all sublists are empty and return 0 if true
        if all(len(sublist) == 0 for sublist in self.feature_selections):
            return 0

        # First, we need to determine the unique features across all datasets
        unique_features = set(
            feature for sublist in self.feature_selections for feature in sublist
        )
        F_f = {feature: 0 for feature in unique_features}

        # Count the occurrences of each feature
        for sublist in self.feature_selections:
            for feature in sublist:
                F_f[feature] += 1

        # Calculate the stability measure
        SH_S = 0
        for f, count in F_f.items():
            if count > 0:
                SH_S += count * np.log2(count)

        if (
            self.n == 1
        ):  # If there is only one subset, return 0 as stability is not meaningful
            return 0

        return SH_S / (self.N * np.log2(self.n))
