Module pytest_util_functions
============================
Initialization file for the 'pytest_util_functions' module, which contains utility functions to be used for writing pytests.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>

Sub-modules
-----------
* pytest_util_functions.assertion_checker
* pytest_util_functions.comparison

Functions
---------

    
`check_assertions(my_val: Any, top_type: Optional[Type] = None, top_length: int | None = None, element_type: Optional[Type] = None, element_length: int | None = None) -> NoneType`
:   Assertion helper function used to check if
    a given input element is of a certain size,
    and its elements are of a certain type and
    size (if needed).
    
    Parameters
    ----------
    my_val : Any
        The value for which assertions will be checked.
    top_type : typing.Type | None, optional
        The type of 'my_val', which will be checked
        if it is equal to None.
        Default: None.
    top_length : int | None, optional
        The length of the value that will be checked,
        if not None.
        Default: None.
    element_type : typing.Type | None, optional
        The expected type of an element of 'my_val',
        which will be checked, if not None.
        Default: None.
    element_length : int | None, optional
        The length of an element of 'my_val',
        which will be checked, if not None.
        Default: None.

    
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