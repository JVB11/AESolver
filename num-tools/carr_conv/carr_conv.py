"""Python library module containing functions that can be used for array conversions of Numpy arrays loaded from GYRE data.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import numpy as np


# utility function used to construct complex numpy arrays
def re_im(val, dim_expr):
    """Generic utility function used to construct complex numpy arrays.

    Parameters
    ----------
    val : np.ndarray or np.void
        The real and imaginary valued array loaded from the GYRE files.
    dim_expr : np.ndarray or np.void or float
        The expression used to perform the dimensioning.

    Returns
    -------
    np.ndarray
        The complex valued array.
    """
    # generate the empty numpy array
    # - rule for void handling
    if isinstance(val, np.void):
        my_complex_array = np.empty((1,), dtype=np.complex128)
    else:
        my_complex_array = np.empty((val['re'].shape[0],), dtype=np.complex128)
    # fill the array
    my_complex_array.real = val['re'] * dim_expr  # type: ignore
    my_complex_array.imag = val['im'] * dim_expr  # type: ignore
    # return the array
    return my_complex_array


# utility function used to construct complex numpy arrays from provided parts
def re_im_parts(real_val, imag_val, check_for_void=False):
    """Generic utility function used to construct a complex numpy array using two parts of a (GYRE) quantity.

    Parameters
    ----------
    real_val : np.ndarray
        The real part of the quantity.
    imag_val : np.ndarray
        The imaginary part of the quantity.
    check_for_void : bool, optional
        If True, check for a void input type. If False, do not perform check; by default False.

    Returns
    -------
    my_complex_array : np.ndarray
        The complex valued array.
    """
    # initialize the complex array (check for void type if needed)
    if (
        check_for_void
        and isinstance(real_val, np.void)
        and isinstance(imag_val, np.void)
    ):
        my_complex_array = np.empty((1,), dtype=np.complex128)
    else:
        my_complex_array = np.empty((real_val.shape[0],), dtype=np.complex128)  # type: ignore
    # fill it
    my_complex_array.real = real_val
    my_complex_array.imag = imag_val
    # return it
    return my_complex_array
