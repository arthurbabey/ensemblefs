from ..core.feature import Feature


class MergingStrategy:
    """
    Base class for merging strategies. This class provides the interface and common functionality
    for different merging strategies.

    Attributes:
        strategy_type (str): Specifies the type of merging strategy, either 'set-based' or 'rank-based'.
    """

    def __init__(self, strategy_type):
        """
        Initializes the MergingStrategy with the specified strategy type.

        Args:
            strategy_type (str): The type of merging strategy ("set-based" or "rank-based").
        """
        self.strategy_type = strategy_type

    def merge(self, data, k_features=None, **kwargs):
        """
        Merges the input data according to the strategy. Must be implemented by subclasses.

        Args:
            data (list): The data to be merged.
            k_features (int, optional): The number of top features to select. Defaults to None.
            **kwargs: Additional keyword arguments specific to the merging strategy.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def is_set_based(self):
        """
        Checks if the strategy is set-based.

        Returns:
            bool: True if the strategy is set-based, False otherwise.
        """
        return self.strategy_type == "set-based"

    def is_rank_based(self):
        """
        Checks if the strategy is rank-based.

        Returns:
            bool: True if the strategy is rank-based, False otherwise.
        """
        return self.strategy_type == "rank-based"

    def _validate_input(self, subsets):
        """
        Validates that the input subsets contain Feature objects.

        Args:
            subsets (list or list of lists): The subsets to validate.

        Raises:
            ValueError: If the subsets are empty, or do not contain Feature objects.
        """
        if not subsets:
            raise ValueError("Subsets cannot be empty")

        # Check if subsets is a list of lists or a single list
        if isinstance(subsets, list):
            if (isinstance(subset[0], list) for subset in subsets):
                # It's a list of lists
                if not subsets or not subsets[0]:  # Check if the first subset is empty
                    raise ValueError("Subsets cannot contain empty lists")
                if not isinstance(subsets[0][0], Feature):
                    raise ValueError("Subsets must contain Feature objects")
            else:
                # It's a single list
                if not isinstance(subsets[0], Feature):
                    raise ValueError("Subsets must contain Feature objects")
        else:
            raise ValueError("Subsets must be a list or a list of lists")
