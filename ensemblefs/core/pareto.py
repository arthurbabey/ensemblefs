from typing import List, Union


class ParetoAnalysis:
    def __init__(self, data: List[List[float]], group_names: List[str]) -> None:
        """
        Initializes the ParetoAnalysis with the provided dataset and group names.

        Args:
            data: A two-dimensional list where each sublist represents the metrics for a group.
            group_names: Names corresponding to each group.
        """
        if not data:
            raise ValueError("Data cannot be empty.")
        self.data: List[List[float]] = data
        self.num_groups: int = len(data)
        self.num_metrics: int = len(data[0])
        self.group_names: List[str] = group_names
        # Each result contains: [group_name, dominate_count, is_dominated_count, scalar score]
        self.results: List[List[Union[str, int]]] = [
            [0] * 4 for _ in range(self.num_groups)
        ]

    def group_dominate_count(self, group_index: int) -> int:
        """
        Calculates the number of groups dominated by the specified group.

        Args:
            group_index: Index of the group in the data list.

        Returns:
            Count of groups that the specified group dominates.
        """
        group = self.data[group_index]
        dominate_count = 0
        for other_index, other_group in enumerate(self.data):
            if other_index == group_index:
                continue
            if all(group[i] >= other_group[i] for i in range(self.num_metrics)) and any(
                group[i] > other_group[i] for i in range(self.num_metrics)
            ):
                dominate_count += 1
        return dominate_count

    def group_isdominated_count(self, group_index: int) -> int:
        """
        Calculates the number of groups that dominate the specified group.

        Args:
            group_index: Index of the group in the data list.

        Returns:
            Count of groups that dominate the specified group.
        """
        group = self.data[group_index]
        dominated_count = 0
        for other_index, other_group in enumerate(self.data):
            if other_index == group_index:
                continue
            if all(group[i] <= other_group[i] for i in range(self.num_metrics)) and any(
                group[i] < other_group[i] for i in range(self.num_metrics)
            ):
                dominated_count += 1
        return dominated_count

    def compute_dominance(self) -> None:
        """
        Computes the dominance scores for all groups. For each group, the scalar score is defined as:
            scalar = dominate_count - is_dominated_count
        """
        for idx in range(self.num_groups):
            self.results[idx][0] = self.group_names[idx]
            dominate = self.group_dominate_count(idx)
            dominated = self.group_isdominated_count(idx)
            self.results[idx][1] = dominate
            self.results[idx][2] = dominated
            self.results[idx][3] = dominate - dominated

    def get_results(self) -> List[List[Union[str, int]]]:
        """
        Returns the sorted results based on the scalar dominance score in descending order.

        Returns:
            Sorted results: each entry includes [group_name, dominate_count, is_dominated_count, scalar score].
        """
        self.compute_dominance()
        sorted_results = sorted(self.results, key=lambda x: x[3], reverse=True)
        return sorted_results
