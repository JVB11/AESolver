'''Python module containing function used to compare dictionaries.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
'''
# import statements
import math
import typing
import numpy as np
# helper types
from mock_gyre_detail import void_type


# comparison function for dictionaries for GYRE Profile
def compare_dicts(first: dict[str, typing.Any],
                  second: dict[str, typing.Any]) -> bool:
    """Compares the contents of two dictionaries.

    Parameters
    ----------
    first : dict[str, typing.Any]
        The first dictionary.
    second : dict[str, typing.Any]
        The second dictionary.
        
    Returns
    -------
    bool
        True if the contents of both dictionaries are comparable, False, if not.
    """
    # check if lengths are the same
    check_len = len(first) == len(second)
    # initialize the comparable bool to True
    # (if multiplied with False, it will become False)
    if check_len:
        comparable = True
        # loop through the first dictionary and compare values with
        # those of the second dictionary
        for _k, _v in first.items():
            if isinstance(_v, np.ndarray):
                if _v.dtype == void_type:
                    # compare both the 're' and 'im' parts
                    my_re = np.allclose(
                        _v['re'], second[_k]['re']
                        )
                    my_im = np.allclose(
                        _v['im'], second[_k]['im']
                        )
                    comparable *= my_re * my_im
                else:
                    comparable *= np.allclose(
                        _v, second[_k]
                        )
            else:
                comparable *= _v == second[_k]
        # if both comparable and lengths are same, return True
        return comparable
    else:
        return False


# comparison function for MESA profile files
def compare_mesa_profile(
    expected_mesa: dict[str, typing.Any],
    vals_mesa: dict[str, typing.Any]
    ) -> bool:
    """Compares MESA profile file dictionaries.

    Parameters
    ----------
    first : dict[str, typing.Any]
        The first dictionary.
    second : dict[str, typing.Any]
        The second dictionary.
        
    Returns
    -------
    bool
        True if the contents of both dictionaries are comparable, False, if not.
    """
    # check if lengths are the same
    check_len = len(expected_mesa) == len(vals_mesa)
    # initialize the comparable bool to True
    # (if multiplied with False, it will become False)
    if check_len:
        comparison_bool = True
        # loop through the first dictionary and compare values with
        # those of the second dictionary
        for _k, (_v, _t) in expected_mesa.items():
            if _t == np.ndarray:
                my_expect = np.frombuffer(_v)
                my_val = np.frombuffer(vals_mesa[_k])
                comparison_bool *= np.allclose(my_expect, my_val)
            elif _t == float:
                my_expect = float(_v)
                my_val = float(vals_mesa[_k])
                comparison_bool *= math.isclose(my_expect, my_val)
            elif _t == int:
                my_expect = int(_v)
                my_val = int(vals_mesa[_k])
                comparison_bool *= math.isclose(my_expect, my_val)
            else:
                comparison_bool *= _v == vals_mesa[_k]
        # return the result from both tests
        return comparison_bool
    else:
        return False
