from typing import Optional


class Feature:
    """Represents a feature with an optional score and selection status."""

    def __init__(
        self, name: str, score: Optional[float] = None, selected: bool = False
    ) -> None:
        """
        Args:
            name: The name of the feature.
            score: The score or importance of the feature (optional).
            selected: Whether the feature has been selected.
        """
        self.name: str = name
        self.score: Optional[float] = score
        self.selected: bool = selected

    def set_score(self, score: float) -> None:
        """Sets the feature score."""
        self.score = score

    def set_selected(self, selected: bool) -> None:
        """Sets the feature selection status."""
        self.selected = selected

    def __str__(self) -> str:
        return (
            f"Feature(name={self.name}, score={self.score}, selected={self.selected})"
        )

    def __repr__(self) -> str:
        return f"Feature('{self.name}', {self.score}, {self.selected})"
