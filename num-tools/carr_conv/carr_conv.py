"""Python library module containing functions that can be used for array conversions of Numpy arrays loaded from GYRE data.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
import numpy as np


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
    # Specific rule for handling void value input
    if isinstance(val, np.void):
        my_complex_array = np.empty((1,), dtype=np.complex128)
    else:
        my_complex_array = np.empty((val['re'].shape[0],), dtype=np.complex128)
    my_complex_array.real = val['re'] * dim_expr  # type: ignore
    my_complex_array.imag = val['im'] * dim_expr  # type: ignore
    return my_complex_array


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
    if (
        check_for_void
        and isinstance(real_val, np.void)
        and isinstance(imag_val, np.void)
    ):
        my_complex_array = np.empty((1,), dtype=np.complex128)
    else:
        my_complex_array = np.empty((real_val.shape[0],), dtype=np.complex128)  # type: ignore
    my_complex_array.real = real_val
    my_complex_array.imag = imag_val
    return my_complex_array
