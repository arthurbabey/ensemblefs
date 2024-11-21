class Feature:
    def __init__(self, name, score=None, selected=False):
        """
        Initialize a new Feature object with a name, optional score, and selection status.

        Args:
            name (str): The name of the feature.
            score (float, optional): The score or importance of the feature. Defaults to None.
            selected (bool, optional): Indicates whether the feature has been selected. Defaults to False.
        """
        self.name = name
        self.score = score
        self.selected = selected

    def set_score(self, score):
        """
        Set the score of the feature.

        Args:
            score (float): The new score to assign to the feature.
        """
        self.score = score

    def set_selected(self, selected):
        """
        Set the selection status of the feature.

        Args:
            selected (bool): The new selection status to assign to the feature.
        """
        self.selected = selected

    def get_name(self):
        """
        Get the name of the feature.

        Returns:
            str: The name of the feature.
        """
        return self.name

    def get_score(self):
        """
        Get the score of the feature.

        Returns:
            float: The score of the feature.
        """
        return self.score

    def get_selected(self):
        """
        Get the selection status of the feature.

        Returns:
            bool: The selection status of the feature.
        """
        return self.selected

    def __str__(self):
        """
        Return a string representation of the feature.

        Returns:
            str: A string that includes the name, score, and selection status of the feature.
        """
        return f"Feature: {self.name}, Score: {self.score}, Selected: {self.selected}"

    def __repr__(self):
        """
        Return an official string representation of the feature.

        Returns:
            str: A formal representation of the feature suitable for debugging.
        """
        return f"Feature({self.name}, {self.score}, {self.selected})"
