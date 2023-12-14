Module pytest_util_functions.comparison
=======================================
Python module containing functions to compare values with each other, as well as iterable elements and dictionaries.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>

Functions
---------

    
`compare_dicts(first: dict, second: dict, len_enforced: bool = True) -> bool`
:   Utility function that compares dictionaries
    and their elements for equivalence.
    
    Parameters
    ----------
    first : dict
        The first dictionary.
    second : dict
        The second dictionary.
    len_enforced : bool, optional
        Whether the length equivalence
        of two dictionaries is enforced.
        Default: True.
    
    Returns
    -------
    bool
        Determines whether the values in
        both dictionaries are the same.

    
`compare_elements(first: dict | list, second: dict | list) -> bool`
:   Compares the elements of two dictionaries.
    
    Parameters
    ----------
    first : dict | list
        The first dictionary or list.
    second : dict | list
        The second dictionary or list.
    
    Returns
    -------
    comparable : bool
        Whether the elements of the first dictionary
        match those elements (with SAME keys) in the
        second dictionary.

    
`value_comparison(first: Any, second: Any) -> bool`
:   Returns whether two values (of any type)
    are equal to each other.
    
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