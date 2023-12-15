"""Python superclass for the classes that compute combinations of different inputs.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""


class GenericCombination:
    """Generic combination-computing superclass."""

    # attribute type declarations
    _value: list
    _cardinality: int

    def __init__(self) -> None:
        pass

    @property
    def c(self) -> int:
        """Returns the cardinality of the resulting combination list.

        Returns
        -------
        int
            Cardinality of the resulting combination list.
        """
        return self._cardinality

    @property
    def l(self) -> list:
        """Returns the resulting combination list.

        Returns
        -------
        list
            Resulting combination list.
        """
        return self._value

    def _compute_combinations(self):
        """To be overloaded by sub-class."""
        pass

    def _compute_cardinality(self):
        """To be overloaded by sub-class."""
        pass
