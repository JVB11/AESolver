Module pytest_util_functions.assertion_checker
==============================================
Python module containing utility function that checks certain assertions about types and lengths of input values.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>

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