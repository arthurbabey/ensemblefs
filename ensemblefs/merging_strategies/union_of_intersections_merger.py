from itertools import combinations

from .base_merger import MergingStrategy


class UnionOfIntersectionsMerger(MergingStrategy):
    """
    A set-based merging strategy that computes the union of intersections of pairs of subsets.
    This class extends the MergingStrategy base class.
    """

    name = "UnionOfIntersections"

    def __init__(self):
        """
        Initializes the UnionOfIntersections strategy with 'set-based' type.
        """
        super().__init__("set-based")

    def merge(self, subsets, num_features=None, **kwargs):
        """
        Merges the input subsets by computing the union of intersections of pairs of subsets.

        Args:
            subsets (list of lists): A list of subsets to be merged.
            **kwargs: Additional keyword arguments specific to the merging strategy.

        Returns:
            set: The union of intersections of the input subsets.

        Raises:
            ValueError: If subset does not contain Feature objects.
        """

        self._validate_input(subsets)
        if len(subsets) == 1:
            return [feature.name for feature in subsets[0]]

        else:
            names = [[feature.name for feature in subset] for subset in subsets]

            pairs = combinations(range(len(names)), 2)
            intersections = [set(names[i]) & set(names[j]) for i, j in pairs]
            return set().union(*intersections)
