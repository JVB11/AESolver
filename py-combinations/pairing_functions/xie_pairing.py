"""Python module containing an implementation of the symmetric Xie binary pairing function and its inverse.

Notes:
------
See https://ui.adsabs.harvard.edu/abs/2021arXiv210510752X/abstract

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import modules
import numpy as np
from math import floor


# accented sign function defined by Xie
def sgn_accent(num):
    """Accented sign function defined in the Xie manuscript.

    Parameters
    ----------
    num : int
        The number argument for the accented sign function.

    Returns
    -------
    int
        The result of the accented sign function applied to 'num'.
    """
    return int((np.abs(np.sign(num)) + np.sign(num)) / 2)


# Xie pairing function
def xie_symmetric_pair(my_pair, lowest_integer):
    """Xie binary pairing function.

    Parameters
    ----------
    my_pair : tuple[int]
        The (unordered) binary pair for which a symmetric mapping is sought.
    lowest_integer : int, optional
        The lowest integer value considered for the pairing. This parameter decreases the value of the mapped integers after transformation; by default 0 (i.e. no lower bound).

    Returns
    -------
    int
        The mapped/paired integer.
    """
    # check that for a tuple (a,b), a < b, otherwise invert (symmetry!)
    if my_pair[0] < my_pair[1]:
        my_pair = my_pair[::-1]
    # adjust radial orders to generate the mapping pair
    map_pair: np.ndarray = np.array(my_pair) - lowest_integer + 1
    # compute intermediate values, following Xie
    int_sum = map_pair.sum() - 1
    minint = int((map_pair.sum() - np.abs(map_pair[0] - map_pair[1])) / 2)
    mod_int = int_sum % 2
    # return the mapping result
    return int(((int_sum * int_sum) - mod_int) / 4) + minint


# inverted Xie pairing function
def invert_xie_symmetric_pair(xie_number, lowest_integer):
    """Inversion of the Xie binary pairing function.

    Parameters
    ----------
    xie_number : int
        The mapped Xie integer.
    lowest_integer : int, optional
        The lowest integer value considered for the pairing. This parameter decreases the value of the mapped integers after transformation; by default 0 (i.e. no lower bound).

    Returns
    -------
    tuple | list
        The tuple/list that was mapped by the Xie pairing function.
    """
    # compute intermediate values
    _t = floor(np.sqrt(xie_number))
    _tsq = _t * _t
    _a = int(np.sign(xie_number - _tsq))
    _b = sgn_accent(xie_number - _tsq - _t)
    _first_fac = (1 - _a) * _t
    # compute the members of the reconstructed pair
    m = _first_fac + _a * ((_t + 1) * (_t + 1 + _b) - xie_number)
    n = _first_fac + _a * (xie_number - (_t * (_t + _b)))
    # return the reconstructed pair
    return (m, n) + lowest_integer - 1
