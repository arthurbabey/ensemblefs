# merging_strategies/__init__.py

from .base_merger import MergingStrategy
from .borda_merger import BordaMerger
from .union_of_intersections_merger import UnionOfIntersectionsMerger

__all__ = [
    "MergingStrategy",
    "BordaMerger",
    "UnionOfIntersectionsMerger",
]
