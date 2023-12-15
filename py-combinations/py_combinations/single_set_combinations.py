"""Python module containing a class that computes the combinations of the elements of a single set.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import modules
from itertools import combinations
from math import comb

# import superclass
from .generic_combination_class import GenericCombination


class SingleSetCombos(GenericCombination):
    """Class that computes the pairwise combinations of the elements in an input set/list/array/range.

    Parameters
    ----------
    my_set : list | set | range | array | np.ndarray
        Set for which pairwise combinations need to be computed.
    """

    # attribute type declarations
    _set: set | list

    def __init__(self, my_set) -> None:
        # initialize the superclass
        super().__init__()
        # store the input list/set/range/array/np.ndarray
        self._set = my_set
        # compute the combinations of elements in the passed set
        self._compute_combinations()
        # compute the cardinality
        self._compute_cardinality()

    def _compute_combinations(self):
        """Computes the combinations of elements in the input set."""
        self._value = list(combinations(self._set, 2))

    def _compute_cardinality(self):
        """Computes the cardinality of the combination set."""
        self._cardinality = comb(len(self._set), 2)
