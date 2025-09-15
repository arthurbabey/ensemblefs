import numpy as np


class ParetoAnalysis:
    """Rank groups by dominance and break ties using utopia distance.

    For each group, computes a scalar dominance score: dominated−is_dominated.
    If the top score ties, scales tied vectors to [0, 1] (within the tie) and
    picks the one closest to the utopia point (1, ..., 1).
    """

    def __init__(self, data: list, group_names: list) -> None:
        """Initialize the analysis state.

        Args:
            data: Metric vectors per group.
            group_names: Display names for groups.

        Raises:
            ValueError: If ``data`` is empty.
        """
        if not data:
            raise ValueError("Data cannot be empty.")
        self.data = data
        self.group_names = group_names
        self.num_groups, self.num_metrics = len(data), len(data[0])

        # Each row will hold:
        #   0  group name
        #   1  dominate_count
        #   2  is_dominated_count
        #   3  scalar = 1 − 2
        #   4  metrics vector  ← NEW column used only for tie-break
        self.results: list = [
            [g, 0, 0, 0, vec]  # vec = data[i]
            for g, vec in zip(group_names, data)
        ]

    def _dominate_count(self, i: int) -> int:
        g = self.data[i]
        return sum(
            all(g[m] >= o[m] for m in range(self.num_metrics)) and any(g[m] > o[m] for m in range(self.num_metrics))
            for j, o in enumerate(self.data)
            if j != i
        )

    def _is_dominated_count(self, i: int) -> int:
        g = self.data[i]
        return sum(
            all(g[m] <= o[m] for m in range(self.num_metrics)) and any(g[m] < o[m] for m in range(self.num_metrics))
            for j, o in enumerate(self.data)
            if j != i
        )

    def get_results(self) -> list:
        """Compute dominance and return ranked rows.

        Returns:
            Rows [name, dominate_count, is_dominated_count, scalar] sorted by rank.
        """
        # 1) scalar dominance
        for i in range(self.num_groups):
            dom = self._dominate_count(i)
            sub = self._is_dominated_count(i)
            self.results[i][1:4] = [dom, sub, dom - sub]

        # 2) initial sort: scalar desc  then lexicographic name
        self.results.sort(key=lambda r: (-r[3], tuple(r[0])))

        # 3) tie-break on utopia distance
        top_scalar = self.results[0][3]
        tied_rows = [r for r in self.results if r[3] == top_scalar]

        if len(tied_rows) > 1:
            tied_data = np.vstack([r[4] for r in tied_rows], dtype=float)

            mins, maxs = tied_data.min(0), tied_data.max(0)
            span = np.where(maxs - mins == 0, 1, maxs - mins)
            scaled = (tied_data - mins) / span  # 0-1 per metric
            dists = np.linalg.norm(1.0 - scaled, axis=1)  # to utopia (1,…,1)

            best_local_idx = int(dists.argmin())  # index inside tied_rows
            best_row = tied_rows[best_local_idx]

            # place best_row at position 0, keep relative order of the rest
            self.results.remove(best_row)
            self.results.insert(0, best_row)

        # strip the metrics vector column before returning (keep original layout)
        return [row[:4] for row in self.results]
