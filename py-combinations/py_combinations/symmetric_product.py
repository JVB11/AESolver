"""Python module containing class that computes generic symmetric products of inputs.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import necessary modules
from itertools import combinations
from math import comb
import logging

# import superclass
from .generic_combination_class import GenericCombination


# initialize logger
logger = logging.getLogger()


class SymmetricProductCombos(GenericCombination):
    """Class that computes a generic set of symmetric pairs of elements of two different sets/lists/arrays/ranges A and B, taking diagonal combinations into account, if necessary.

    Parameters
    ----------
    A : set | list | array | np.ndarray | range
        First of the two sets for which combinations need to be computed.
    B : set | list | array | np.ndarray | range
        Second of the two sets for which combinations need to be computed.
    include_diagonal : bool, optional
        Defines whether elements on the diagonal should be taken into account when generating the list of symmetric pairings; by default False.
    """

    # attribute type declarations
    _A: set
    _B: set
    _A_combs: set
    _B_combs: set
    include_diagonal: bool
    _diagonal_cardinality: int

    def __init__(self, A, B, include_diagonal=False) -> None:
        # initialize the super-class
        super().__init__()
        # convert input to sets of elements
        try:
            self._A = set(A)
            self._B = set(B)
        except TypeError:
            logger.exception(
                f'Input is not iterable and cannot be converted to a set. (A: {A}, B: {B}, include_diagonal: {include_diagonal}) Now exiting.'
            )
        # compute combinations of elements in individual input sets/lists/arrays
        self._A_combs = set(combinations(A, 2))
        self._B_combs = set(combinations(B, 2))
        # store whether you want to include overlapping elements
        self.include_diagonal = include_diagonal
        # compute the symmetric product of combinations
        self._compute_combinations()
        # compute the cardinality of the product
        self._compute_cardinality()
        # account for the diagonal input between input sets, if necessary
        self._account_for_diagonal()

    def _compute_combinations(self):
        """Compute the generic set of symmetric pairs of elements of two sets/lists/arrays, taking any overlap into account."""
        # compute auxiliary variables
        # - pairs of elements in union of input sets/lists/arrays
        _A_union_B_combs = set(combinations(self._A | self._B, 2))
        # - pairs of elements that uniquely exist in the input sets/lists/arrays
        _A_min_B = set(combinations(self._A - self._B, 2))
        _B_min_A = set(combinations(self._B - self._A, 2))
        # compute different terms that contribute to the list of symmetric pairings between two sets
        _first_term = _A_union_B_combs - (self._A_combs | self._B_combs)
        _second_term = self._A_combs - _A_min_B
        _third_term = self._B_combs - _B_min_A
        # store the list of symmetric pairs of elements that are not in the union of the two input sets
        self._value = list(_first_term | _second_term | _third_term)

    def _compute_cardinality(self):
        """Compute the cardinality of the symmetric product set."""
        # compute the cardinality of the overlapping elements in the sets
        self._diagonal_cardinality = len(list(self._A & self._B))
        # compute the cardinalities of the individual sets
        card_A = len(self._A)
        card_B = len(self._B)
        # compute the different terms that contribute to the overall cardinality of the symmetric product
        card_1 = comb(card_A + card_B - self._diagonal_cardinality, 2)
        card_2 = comb(card_A - self._diagonal_cardinality, 2)
        card_3 = comb(card_B - self._diagonal_cardinality, 2)
        # compute and store the overall cardinality
        self._cardinality = card_1 - card_2 - card_3

    def _account_for_diagonal(self):
        """Account for the diagonal between the two input sets and add these pairs to the result, if necessary."""
        if self.include_diagonal:
            # generate the list of diagonal element pairings and add it to the return _value
            self._value += [
                (diagonal_element, diagonal_element)
                for diagonal_element in (self._A & self._B)
            ]
            # correct the cardinality for the inclusion of diagonal element pairings
            self._cardinality += self._diagonal_cardinality
