"""Python module containing an implementation of a symmetric binary pairing function and its inverse.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import modules
import numpy as np
from math import floor


# implementation of a simplistic symmetric pairing function
def symmetric_pair(my_pair, lowest_integer=0):
    """Simplistic symmetric binary pairing function.

    Parameters
    ----------
    my_tup : tuple[int]
        Contains the pair of integers to be mapped (symmetrically).
    lowest_integer : int, optional
        The lowest integer value considered for the pairing. This parameter decreases the value of the mapped integers after transformation; by default 0 (i.e. no lower bound).

    Returns
    -------
    int
        The mapped/paired integer.
    """
    # check that for a tuple (a,b), a >= b, otherwise invert (symmetry!)
    if my_pair[0] >= my_pair[1]:
        my_pair = my_pair[::-1]
    # compute the members of the mapping pair
    a, b = np.array(my_pair) - lowest_integer + 1
    # return the mapped integer
    return a + int((b * (b - 1)) / 2)


# inversion of the simplistic symmetric pairing function
def invert_symmetric_pair(my_sym_number, lowest_integer=0):
    """Inverts the simplistic symmetric binary pairing number mapping.

    Parameters
    ----------
    my_sym_number : int
        The mapped/paired integer.
    lowest_integer : int, optional
        The lowest integer value considered for the pairing. This parameter decreases the value of the mapped integers after transformation; by default 0 (i.e. no lower bound).

    Returns
    -------
    tuple or list
        The tuple/list that was mapped by the simplistic symmetric pairing function.
    """
    # compute intermediate quantities
    _w = floor(0.5 * np.sqrt(8.0 * my_sym_number + 1.0) - 0.5)
    _t = int(((_w * _w) + _w) / 2)
    # compute the paired integers
    reconstructed_tuple = (my_sym_number - _t, _w + 1)
    # return the previously mapped tuple
    return reconstructed_tuple + lowest_integer - 1
