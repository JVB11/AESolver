"""Python enumeration file that contains the enumeration class that handles Numba stencils used for numerical differentiation.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
import logging
import sys
from enum import Enum
from numba import stencil  # type: ignore


logger = logging.getLogger(__name__)


# Numba stencil definitions for numerical differentiation
# -------------------------------------------------------
@stencil
def forward_difference_kernel_first_order(x, y):
    return (y[1] - y[0]) / (x[1] - x[0])


@stencil
def backward_difference_kernel_first_order(x, y):
    return (y[0] - y[-1]) / (x[0] - x[-1])


@stencil
def center_kernel_first_order(x, y):
    return (y[1] - y[-1]) / (x[1] - x[-1])


@stencil
def center_kernel_second_order(x, y):
    return (-1.0 * y[2] + 8.0 * y[1] - 8.0 * y[-1] + y[-2]) / (
        12 * (x[1] - x[0])
    )


class StencilFunctions(Enum):
    """Enumeration class used to store Numba finite differencing stencil methods.

    Notes
    -----
    Finite difference stencil of order greater than 1 use approximated step functions in the 'x'-variable.
    """

    # declare the enumeration (dictionary) members holding the stencil methods
    # - the keywords are strings that define the order of the kernel accuracy
    FORWARD = {'1': forward_difference_kernel_first_order, '2': None}
    BACKWARD = {'1': backward_difference_kernel_first_order, '2': None}
    CENTER = {'1': center_kernel_first_order, '2': center_kernel_second_order}

    @classmethod
    def retrieve_stencil(cls, my_order=1, my_differencing_type='center'):
        """Class method used to retrieve the stencil function.

        Parameters
        ----------
        my_order : int, optional
            Finite differencing order. Default: 1 (first order differencing).
        my_differencing_type : str, optional
            Finite differencing stencil type, by default 'center'. Choice between:

            * 'forward': defines a forward differencing function.
            * 'backward': defines a backward differencing function.
            * 'center': defines a centralized differencing function.

        Returns
        -------
        stencil_func: Callable
            The result of the finite differencing by the Numba kernel.
        str
            Information on the numerical derivative models.
        """
        _stencil_dict = cls.__getitem__(my_differencing_type.upper()).value
        try:
            return (
                _stencil_dict[f'{my_order}'],
                f'Numba {my_differencing_type} stencil models of order '
                f'{my_order} valid for first derivatives',
            )
        except KeyError:
            logger.exception(
                '\n\nNo useful stencil function can be found. '
                'Check your input! Now exiting.\n\n\n'
            )
            sys.exit()
