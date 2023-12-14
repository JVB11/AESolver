"""Python module containing utility function that checks certain assertions about types and lengths of input values.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import typing


# define checking function
def check_assertions(
    my_val: typing.Any,
    top_type: typing.Type | None = None,
    top_length: int | None = None,
    element_type: typing.Type | None = None,
    element_length: int | None = None,
) -> None:
    """Assertion helper function used to check if a given input element is of a certain size, and its elements are of a certain type and size (if needed).

    Parameters
    ----------
    my_val : Any
        The value for which assertions will be checked.
    top_type : typing.Type | None, optional
        The type of 'my_val', which will be checked if it is equal to None; by default None.
    top_length : int | None, optional
        The length of the value that will be checked, if not None; by default None.
    element_type : typing.Type | None, optional
        The expected type of an element of 'my_val', which will be checked, if not None; by default None.
    element_length : int | None, optional
        The length of an element of 'my_val', which will be checked, if not None; by default None.
    """
    # check if we need to check value type
    if top_type:
        assert isinstance(my_val, top_type)
    # check if we need to check value length
    if top_length:
        assert len(my_val) == top_length
    # check if we need to check element type or element_length
    if (element_type is not None) or (element_length is not None):
        for my_el in my_val:
            if element_type:
                assert isinstance(my_el, element_type)
            if element_length:
                assert len(my_el) == element_length
