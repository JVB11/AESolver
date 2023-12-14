"""Python module containing functions to compare values with each other, as well as iterable elements and dictionaries.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import math
import typing
import numpy as np
import traceback
import logging


def value_comparison(first: typing.Any, second: typing.Any) -> bool:
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
            try:
                # lambda (or regular?) functions
                value_check *= first.__code__.co_code == second.__code__.co_code
            except AttributeError:
                try:
                    # lists of lambda (or regular functions)
                    value_check *= len(first) == len(second)
                    for _i, _j in zip(first, second):
                        value_check *= (
                            _i.__code__.co_code == _j.__code__.co_code
                        )
                except AttributeError:
                    # lists or None type objects
                    # --> direct comparison
                    value_check *= first == second
                except Exception:  # CATCH ALL OTHER
                    logging.error(traceback.format_exc())
                    value_check = False
            except Exception:  # CATCH ALL OTHER
                logging.error(traceback.format_exc())
                value_check = False
        except Exception:  # CATCH ALL OTHER
            logging.error(traceback.format_exc())
            value_check = False
    except Exception:  # CATCH ALL OTHER
        logging.error(traceback.format_exc())
        value_check = False
    # return the value check value
    return value_check  # type: ignore


def compare_elements(first: dict | list, second: dict | list) -> bool:
    """Compares the elements of two dictionaries.

    Parameters
    ----------
    first : dict | list
        The first dictionary or list.
    second : dict | list
        The second dictionary or list.

    Returns
    -------
    comparable : bool
        Whether the elements of the first dictionary match those elements (with SAME keys) in the second dictionary.
    """
    # initialize the comparison bool
    comparable = True
    # assess comparability
    try:
        # compare the dictionary elements
        for _k, _v in first.items():  # type: ignore
            # get the element in the second dictionary
            _sec_v = second[_k]
            # compare the elements
            comparable *= value_comparison(_v, _sec_v)
    except AttributeError:
        # ensure size of lists is same
        comparable *= len(first) == len(second)
        # compare the list elements
        for _v1, _v2 in zip(first, second):
            # compare elements
            comparable *= value_comparison(_v1, _v2)
    except Exception:  # CATCH ALL OTHER
        logging.error(traceback.format_exc())
        comparable = False
    # return the element comparison bool
    return comparable  # type: ignore


def compare_dicts(first: dict, second: dict, len_enforced: bool = True) -> bool:
    """Utility function that compares dictionaries and their elements for equivalence.

    Parameters
    ----------
    first : dict
        The first dictionary.
    second : dict
        The second dictionary.
    len_enforced : bool, optional
        Whether the length equivalence of two dictionaries is enforced; by default True.

    Returns
    -------
    bool
        Determines whether the values in both dictionaries are the same.
    """
    # check if lengths are the same, if necessary
    if len_enforced:
        len_bool = len(first) == len(second)
        # no comparison made if lengths differ!
        if not len_bool:
            return False
    # compare the dictionary elements (after checking if the length is OK, when requested)
    comparable = compare_elements(first=first, second=second)
    # return the boolean that determines whether the two dictionaries are comparable
    return comparable
