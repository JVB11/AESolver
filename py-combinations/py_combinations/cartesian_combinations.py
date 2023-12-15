"""Python module containing class that computes a Cartesian product of two sets, which contains pairs of the elements in both sets.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import modules
from itertools import product

# import superclass
from .generic_combination_class import GenericCombination


class CartesianCombos(GenericCombination):
    """Class that computes the pairs of elements of two sets using a Cartesian product.

    Parameters
    ----------
    set_A : list | set | range | array | np.ndarray
        The first input set for the cartesian product.
    set_B : list | set | range | array | np.ndarray
        The second input set for the cartesian product.
    """

    # attribute type declarations
    _A: set | list
    _B: set | list

    def __init__(self, set_A, set_B) -> None:
        # initialize the superclass
        super().__init__()
        # store the input lists/sets/ranges/arrays/np.ndarrays
        self._A = set_A
        self._B = set_B
        # compute the combinations of elements in the passed set
        self._compute_combinations()
        # compute the cardinality
        self._compute_cardinality()

    def _compute_combinations(self) -> None:
        """Computes the combinations of elements in the input sets."""
        self._value = list(product(self._A, self._B))

    def _compute_cardinality(self) -> None:
        """Computes the cardinality of the combination set."""
        self._cardinality = len(self._A) * len(self._B)
