from collections import Counter, defaultdict
from itertools import chain
from typing import List, Optional, Set

import numpy as np

from ..core.feature import Feature
from .base_merger import MergingStrategy


class ConsensusMerger(MergingStrategy):
    """
    Set-based merger that keeps features selected by at least `k`
    base selectors (global consensus ≥ k). If `fill=True`, the output
    is trimmed / padded to exactly `num_features_to_select` by using
    the summed, min-max–normalised scores as a tie-breaker.
    """

    def __init__(self, k: int = 2, *, fill: bool = False) -> None:
        super().__init__("set-based")
        self.k = k
        self.fill = fill
        self.name = f"Consensus_ge_{k}"

    # -----------------------------------------------------------------
    def merge(
        self,
        subsets: List[List[Feature]],
        num_features_to_select: Optional[int] = None,
        **kwargs,
    ) -> Set[str]:
        self._validate_input(subsets)

        if self.fill and num_features_to_select is None:
            raise ValueError("`num_features_to_select` required when fill=True")

        # ── collect names & scores (ragged‑safe) ─────────────────────
        names_mat = [[f.name for f in s] for s in subsets]

        # Consensus counts across all selectors
        counts = Counter(chain.from_iterable(names_mat))

        # Summed, per-selector min‑max–normalised scores per feature name
        sum_scores = defaultdict(float)
        for subset in subsets:
            if not subset:
                continue
            scores = np.array([f.score for f in subset], dtype=np.float32)
            min_v = float(scores.min())
            rng = float(scores.max() - min_v) or 1.0
            norm = (scores - min_v) / rng
            for name, s in zip([f.name for f in subset], norm):
                sum_scores[name] += float(s)

        selected = {f for f, c in counts.items() if c >= self.k}

        if not self.fill:
            return selected

        # ── trim / pad to desired size ───────────────────────────────
        core_sorted = sorted(selected, key=lambda n: sum_scores[n], reverse=True)
        if len(core_sorted) >= num_features_to_select:
            return set(core_sorted[:num_features_to_select])

        extras = sorted(
            (n for n in counts if n not in selected),
            key=lambda n: sum_scores[n],
            reverse=True,
        )
        need = num_features_to_select - len(core_sorted)
        return set(core_sorted + extras[:need])
