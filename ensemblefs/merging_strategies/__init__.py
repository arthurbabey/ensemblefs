# merging_strategies/__init__.py

from .base_merger import MergingStrategy
from .borda_merger import BordaMerger
from .union_of_intersections_merger import UnionOfIntersectionsMerger
from .arithmetic_mean_merger import ArithmeticMeanMerger
from .l2_norm_merger import L2NormMerger
from .consensus_merger import ConsensusMerger

__all__ = [
    "MergingStrategy",
    "BordaMerger",
    "UnionOfIntersectionsMerger",
    "ArithmeticMeanMerger",
    "L2NormMerger",
    "ConsensusMerger",
]
