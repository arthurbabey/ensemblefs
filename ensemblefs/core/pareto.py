class ParetoAnalysis:
    def __init__(self, data, group_names):
        """
        Initialize the ParetoAnalysis with the provided dataset and group names.

        Args:
            data (list of lists): A two-dimensional list where each sub-list represents
                                  the metrics for a group.
            group_names (list of str): Names corresponding to each group in the data.

        Attributes:
            num_groups (int): The number of groups in the dataset.
            num_metrics (int): The number of metrics per group.
            is_dominated_freq (list of int): List tracking how often each group is dominated.
            dominated_freq (list of int): List tracking how often each group dominates others.
            results (list of lists): Stores results including group names, dominate count,
                                     is dominated count, and a scalar score calculated
                                     from dominate and is dominated counts.
        """
        self.data = data
        self.num_groups = len(data)
        self.num_metrics = len(data[0])
        self.is_dominated_freq = [0] * self.num_metrics
        self.dominated_freq = [0] * self.num_metrics
        self.group_names = group_names
        # 4 columns: group_id, dominate_freq, is_dominated_freq, scalar
        self.results = [[0] * 4 for _ in range(self.num_groups)]

    def group_dominate_count(self, group_index):
        """
        Calculate how many groups the specified group dominates.

        Args:
            group_index (int): The index of the group in the data list.

        Returns:
            int: The count of groups that the specified group dominates.
        """
        group = self.data[group_index]
        dominate_count = 0
        for other_group_index, other_group in enumerate(self.data):
            if other_group_index != group_index:
                if all(
                    group[metric_index] >= other_group[metric_index]
                    for metric_index in range(self.num_metrics)
                ) and any(
                    group[metric_index] > other_group[metric_index]
                    for metric_index in range(self.num_metrics)
                ):
                    dominate_count += 1
        return dominate_count

    def group_isdominated_count(self, group_index):
        """
        Calculate how many groups dominate the specified group.

        Args:
            group_index (int): The index of the group in the data list.

        Returns:
            int: The count of groups that dominate the specified group.
        """
        group = self.data[group_index]
        is_dominated_count = 0
        for other_group_index, other_group in enumerate(self.data):
            if other_group_index != group_index:
                if all(
                    group[metric_index] <= other_group[metric_index]
                    for metric_index in range(self.num_metrics)
                ) and any(
                    group[metric_index] < other_group[metric_index]
                    for metric_index in range(self.num_metrics)
                ):
                    is_dominated_count += 1
        return is_dominated_count

    def compute_dominance(self):
        """
        Compute the dominance scores for all groups in the dataset.
        """
        for group_index in range(self.num_groups):
            self.results[group_index][0] = self.group_names[group_index]
            self.results[group_index][1] = self.group_dominate_count(group_index)
            self.results[group_index][2] = self.group_isdominated_count(group_index)
            self.results[group_index][3] = (
                self.results[group_index][1] - self.results[group_index][2]
            )

    def get_results(self):
        """
        Get the sorted results based on the scalar dominance score.

        Returns:
            list of lists: Sorted results including group names, dominate count, is dominated count,
                           and scalar dominance score, sorted by scalar score in descending order.
        """
        self.compute_dominance()
        sorted_results = sorted(self.results, key=lambda x: x[3], reverse=True)
        return sorted_results
