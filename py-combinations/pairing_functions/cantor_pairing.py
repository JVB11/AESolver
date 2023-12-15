"""Python module containing an implementation of the Cantor binary pairing function and its inverse.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import modules
from math import floor
import numpy as np


# Cantor pairing function
def cantor_pair(my_pair, lowest_integer=0):
    """Cantor pairing function that maps a pair of integers bijectively onto a (unique) integer.

    Parameters
    ----------
    my_pair : tuple[int]
        Contains the pair of integers to be mapped.
    lowest_integer : int, optional
        The lowest integer value considered for the pairing. This parameter decreases the value of the mapped integers after transformation; by default 0 (i.e. no lower bound).

    Returns
    -------
    integer
        The paired/mapped integer.
    """
    # adjust the mapping pair: subtract the lowest integer considered
    mapping_pair = my_pair - lowest_integer
    # compute the sum of the mapping integer pair
    int_sum = mapping_pair.sum()
    # compute and return the paired/mapped integer
    return int((int_sum * (int_sum + 1)) / 2) + mapping_pair[1]


# inverse of the Cantor pairing function
def invert_cantor_pair(cantor_integer, lowest_integer=0):
    """Computes the inverse of the Cantor pairing function. This retrieves the pair of integers that were mapped onto the Cantor integer.

    Parameters
    ----------
    cantor_integer : int
        The paired/mapped Cantor integer.
    lowest_integer : int, optional
        The lowest integer value considered for the pairing. This parameter is used to reconstruct the integer pair from the Cantor integers; by default 0 (i.e. no lower radial order bound).

    Returns
    -------
    tuple or list
        The pairwise combination corresponding to the mapped Cantor integer.
    """
    # For a tuple (x,y), w = x + y, t = (w^2 + w) / 2, cantor_integer = t + y
    # - intermediate variable definitions
    _w = floor(0.5 * np.sqrt(8.0 * cantor_integer + 1.0) - 0.5)
    _t = int(((_w * _w) + _w) / 2)
    # - tuple generation
    reconstructed_tuple = (_w - cantor_integer + _t, cantor_integer - _t)
    # - return result
    return reconstructed_tuple + lowest_integer  # type: ignore
