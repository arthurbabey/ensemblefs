import numpy as np
from ranky import borda

from .base_merger import MergingStrategy


class BordaMerger(MergingStrategy):
    """
    A rank-based merging strategy that uses the Borda count method to merge scores.
    This class extends the MergingStrategy base class.
    """

    name = "Borda"

    def __init__(self, **kwargs):
        """
        Initializes the BordaStrategy with 'rank-based' type.
        """
        super().__init__("rank-based")
        self.kwargs = kwargs

    def merge(self, subsets, k_features=None):
        """
        Merges the given subsets using the Borda count method.

        Parameters:
        - subsets (list of list of Feature): The subsets to merge. Each subset must contain Feature objects.
        - k_features (int, optional): The number of top features to return. If None, all features are returned.
        - **kwargs: Additional keyword arguments to pass to the Borda count method.

        Returns:
        - list of str: The names of the top k_features features, sorted by their merged Borda scores.
        """

        self._validate_input(subsets)

        if len(subsets) == 1:
            return [feature.name for feature in subsets[0]]
        else:

            # Extract feature names and scores from the subsets
            feature_names = [
                feature.name for feature in subsets[0]
            ]  # extract feature names mapping from the first subset
            scores = np.array(
                [[feature.score for feature in subset] for subset in subsets]
            ).T
            scores_merged = np.array(borda(m=scores, **self.kwargs))

            # Sort based on Borda scores, low scores means low rank means high importance
            sorted_indices = np.argsort(scores_merged, kind="stable")
            sorted_names = [feature_names[i] for i in sorted_indices]
            return sorted_names[:k_features] if k_features is not None else sorted_names
