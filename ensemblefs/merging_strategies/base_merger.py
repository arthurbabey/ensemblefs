from typing import List, Union

from ..core.feature import Feature


class MergingStrategy:
    """Base class for merging strategies."""

    def __init__(self, strategy_type: str) -> None:
        """
        Args:
            strategy_type: The type of merging strategy ('set-based' or 'rank-based').
        """
        self.strategy_type = strategy_type

    def merge(self, data: List, num_features_to_select: int, **kwargs) -> List[Feature]:
        """
        Merges input data according to the strategy (to be implemented by subclasses).

        Args:
            data: The data to merge.
            num_features_to_select: Number of top features to select.
            **kwargs: Additional strategy-specific arguments.

        Raises:
            NotImplementedError: If not implemented in a subclass.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def is_set_based(self) -> bool:
        """Returns True if the strategy is set-based."""
        return self.strategy_type == "set-based"

    def is_rank_based(self) -> bool:
        """Returns True if the strategy is rank-based."""
        return self.strategy_type == "rank-based"

    def _validate_input(
        self, subsets: Union[List[Feature], List[List[Feature]]]
    ) -> None:
        """
        Validates that subsets contain Feature objects.

        Args:
            subsets: A list or list of lists containing Feature objects.

        Raises:
            ValueError: If subsets are empty or contain invalid types.
        """
        if not subsets:
            raise ValueError("Subsets cannot be empty.")

        if isinstance(subsets[0], list):  # List of lists case
            if not all(isinstance(sub, list) and sub for sub in subsets):
                raise ValueError("Subsets cannot contain empty lists.")
            if not all(
                isinstance(feature, Feature) for sub in subsets for feature in sub
            ):
                raise ValueError("Subsets must contain Feature objects.")
        else:  # Single list case
            if not all(isinstance(feature, Feature) for feature in subsets):
                raise ValueError("Subsets must contain Feature objects.")
