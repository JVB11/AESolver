'''Python module containing functions to compare output dictionaries with expected dictionary values.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
'''
# import statements
import math
import typing
import numpy as np
import traceback
import logging


def _value_comparison(first: typing.Any, second: typing.Any) -> bool:
    """Returns whether two values (of any type) are equal to each other.

    Parameters
    ----------
    first : typing.Any
        The first value.
    second : typing.Any
        The second value.

    Returns
    -------
    value_check : bool
        Whether the two values are equal to each other.
    """
    # initialize the value check boolean
    value_check = True
    # attempt to obtain a dtype (to check if it is a numpy array)
    try:
        # numpy arrays have a dtype
        value_check *= first.dtype == second.dtype
        # numpy arrays need to be checked with a allclose method
        value_check *= np.allclose(first, second)
    except AttributeError:
        # try running a math.isclose operation
        try:
            # works for ints and floats
            value_check *= math.isclose(first, second)
            # AND a type check
            value_check *= type(first) == type(second)
        except TypeError:
            # lists or None type objects --> direct comparison
            value_check *= first == second
        except Exception as d:  # CATCH ALL OTHER
            logging.error(traceback.format_exc())
            value_check = False
    except Exception as e:  # CATCH ALL OTHER
        logging.error(traceback.format_exc())
        value_check = False
    # return the value check value
    return value_check
    

def _compare_elements(first: dict[str, typing.Any],
                      second: dict[str, typing.Any]) -> bool:
    """Compares the elements of two dictionaries.

    Parameters
    ----------
    first : dict[str, typing.Any]
        The first dictionary.
    second : dict[str, typing.Any]
        The second dictionary.

    Returns
    -------
    comparable : bool
        Whether the elements of the first dictionary match those elements (with SAME keys) in the second dictionary.
    """
    # compare the dictionary elements
    comparable = True
    for _k, _v in first.items():
        # get the element in the second dictionary
        _sec_v = second[_k]
        # compare the elements
        comparable *= _value_comparison(_v, _sec_v)
    # return the element comparison bool
    return comparable


def compare_dicts(first: dict[str, typing.Any],
                  second: dict[str, typing.Any],
                  len_enforced: bool=True) -> bool:
    """Utility function that compares dictionaries and their elements for equivalence.

    Parameters
    ----------
    first : dict[str, typing.Any]
        The first dictionary.
    second : dict[str, typing.Any]
        The second dictionary.
    len_enforced : bool, optional
        Whether the length equivalence of two dictionaries is enforced; by default True.

    Returns
    -------
    bool
        Determines whether the values in
        both dictionaries are the same.
    """
    # check if lengths are the same, if necessary
    if len_enforced:
        len_bool = len(first) == len(second)
        # no comparison made if lengths differ!
        if not len_bool: return False 
    # compare the dictionary elements
    comparable = _compare_elements(
        first=first, second=second
        )
    # return the boolean that determines whether
    # the two dictionaries are comparable
    if len_enforced:
        return len_bool and comparable 
    else:
        return comparable
    