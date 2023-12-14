"""Python file containing classes that handle numerical differentiation.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import logging
import sys
import numpy as np
from typing import Callable
from scipy.interpolate import interp1d

# import local custom modules, if Numba is installed
try:
    from .stencils import StencilFunctions

    numba_allowed = True
except ImportError:
    numba_allowed = False
# import custom modules
from myfnd import fnd  # type: ignore


# set up logger
logger = logging.getLogger(__name__)


# class definition for the class containing the numerical differentiation methods
class NumericalDifferentiator:
    """Python class that contains several methods that can be used for numerical differentiation.

    Parameters
    ----------
    order_derivative : int
        The order up to which derivatives shall be computed (zeroth to 'order_derivative'-th order).
    differentiation_method : str, optional
        Denotes the method that will be used for numerical differentiation up to order 'order_derivative'; by default 'gradient'. Options are:

        * 'stencil_functions': uses Numba Stencil functions to derive lower order approximations to the finite differencing estimates.
        * 'gradient': uses Numpy gradient function to estimate the center finite difference approximation of the first derivative.
        * 'fornberg': uses the Lagrange interpolation polynomials of Fornberg (1988) to approximate the numerical derivatives.

    perform_interpolation : bool, optional
        Denotes whether an internal linear interpolation grid will be used to increase number of samples of the input arrays. The output will be provided for the specific values of the input arrays only! This should be used for the 'fornberg' differentiation method if there are few data points in your input arrays; by default False.
    interpolation_factor : int, optional
        Denotes the factor with which sampling will be increased if 'perform_interpolation' is True; by default 10. (A value of e.g. 10 will increase the sampling tenfold!)
    """

    # type hints
    _allowed_types: dict[str, Callable]
    _differentiation_executable: Callable
    _order_derivative: int
    _perform_interpolation: bool
    _interpolation_factor: int

    # initialization
    def __init__(
        self,
        order_derivative,
        differentiation_method='gradient',
        perform_interpolation=False,
        interpolation_factor=10,
    ) -> None:
        # construct the dictionary of allowed numerical differentiation types, which refer to the differentiation methods contained in this class, and booleans that indicate whether smoothing can be applied.
        if numba_allowed:
            self._allowed_types = {
                'stencil_functions': self._finite_difference_stencil,
                'gradient': self._gradient_stencil_numpy,
                'fornberg': self._fornberg_interpolation,
            }
        else:
            self._allowed_types = {
                'gradient': self._gradient_stencil_numpy,
                'fornberg': self._fornberg_interpolation,
            }
        # load the differentiation method
        self._differentiation_executable = self._select_differentiation_method(
            differentiation_method
        )
        # store the order up to which derivatives should be computed, if possible
        self._order_derivative = order_derivative
        # store whether internal linear interpolation should be used to increase the number of samples in the input arrays.
        self._perform_interpolation = perform_interpolation
        # store the interpolation factor
        # NEEDS TO BE AN INTEGER
        if isinstance(interpolation_factor, int):
            self._interpolation_factor = interpolation_factor
        else:
            logger.error(
                'Interpolation factor should be integer!' ' Now exiting.'
            )
            sys.exit()

    ##################################################
    # method that selects the differentiation method #
    ##################################################
    def _select_differentiation_method(self, method_identifier):
        """Internal method used to select the numerical differentiation method.

        Parameters
        ----------
        method_identifier : str
            String that denotes the method that will be used for numerical differentiation up to order 'order_derivative'. Options are:

            * unispline: uses Scipy Univariate splines of a certain degree ('degree_spline') and with a smoothing factor ('smoothing_factor_spline').
            * stencil_functions: uses Numba stencil functions to a certain degree to approximate the first derivative.
            * gradient: uses Numpy gradient function to approximate the first derivative.
            * fornberg: uses the Lagrange interpolation polynomials of Fornberg (1988) to approximate the numerical derivatives.

        Returns
        -------
        Callable
            The differentiation method.
        """
        try:
            return self._allowed_types[method_identifier]
        except KeyError:
            logger.exception(
                'Cannot access the selected differentiation method. Check your input method identifier! Now exiting.'
            )
            sys.exit()

    ###########################################################################
    # internal methods that define how numerical differentiation is performed #
    ###########################################################################
    def _fornberg_interpolation(
        self, x, y, original_formulation=False, interpolation_order=4
    ):
        """Internal method that uses the Lagrange interpolation polynomials defined by Fornberg (1988) to numerically approximate derivatives.

        Parameters
        ----------
        x: np.ndarray
            The x-coordinates from which the derivative model is constructed. These need not be equally spaced.
        y: np.ndarray
            The y-coordinates from which the derivative model is constructed. These need not be equally spaced.
        original_formulation: bool, optional
            If True, use the original Fornberg (1988) algorithm to compute interpolation weights, after which Numpy is used to reconstruct the interpolation polynomials. If False, use the Cocubed pure Fortran implementation of the Fornberg (1988) algorithm that generates the numerical approximation of the derivatives; by default False.
        interpolation_order: int, optional
            The order of accuracy of the Lagrange interpolation scheme used by Fornberg (1988); by default 4.

        Output
        ------
        list[np.ndarray]
            The list containing the Lagrange interpolations models for the derivatives.
        str
            Information on the Lagrange interpolation models.
        """
        # Choose between the two different implementations
        if original_formulation:
            # - Original formulation + numpy implementation
            # TODO: implement this
            sys.exit()
        else:
            # - Cococubed implementation, see: https://cococubed.com/code_pages/fdcoef.shtml
            # initialize the result arrays + fill them
            results = []
            for _ord_der in range(self._order_derivative + 1):
                # append the zero-th derivative
                if _ord_der == 0:
                    results.append(y)
                else:
                    # initialize numpy array in Fortran format
                    _array_der = np.asfortranarray(
                        np.empty_like(y, dtype=np.float64)
                    )
                    # compute the grid derivative
                    fnd.fornberg_grid_derivative(
                        _ord_der, interpolation_order, x, y, _array_der
                    )
                    # append the results (re-conversion to C-contiguity not needed for 1D arrays)
                    results.append(np.ascontiguousarray(_array_der))
            # return the results + information
            return (
                results,
                f'Lagrange interpolation polynomial model based on Fornberg (1988); Cocubed implementation (interpolation accuracy order: {interpolation_order}, derivatives up to order {self._order_derivative}).',
            )

    def _finite_difference_stencil(
        self, x, y, my_order=1, my_differencing_type='center'
    ):
        """Internal method used to obtain a Numba finite difference stencil model used to approximate the first derivative of a set of specific x- and y-coordinates. The finite differencing order, as well as the finite difference stencil type can be adjusted.

        Notes
        -----
        This method does not handle the boundary points of the derivative well: it sets them to zero.

        Parameters
        ----------
        x: np.ndarray
            The x-coordinates from which the derivative model is constructed. These need not be equally spaced.
        y: np.ndarray
            The y-coordinates from which the derivative model is constructed. These need not be equally spaced.
        my_order : int, optional
            Finite differencing order, by default 1 (first order differencing).
        my_differencing_type : str, optional
            Finite differencing stencil type, by default 'center'. Choice between:

            * 'forward': defines a forward differencing function;
            * 'backward': defines a backward differencing function;
            * 'center': defines a centralized differencing function.

        Output
        ------
        list[np.ndarray]
            The list containing the finite difference stencil models for the first derivative.
        str
            Information on the finite difference stencil models.
        """
        # create an alias for the finite differencing method, and retrieve the information string
        _fd_method, info_string = StencilFunctions.retrieve_stencil(
            my_order=my_order, my_differencing_type=my_differencing_type
        )
        # retrieve list of finite difference stencil models
        results = [
            _fd_method(x, y)
            if (_ord_der == 1)
            else y
            if (_ord_der == 0)
            else None
            for _ord_der in range(self._order_derivative + 1)
        ]
        # return the finite difference stencil models and the information string
        return results, info_string

    def _gradient_stencil_numpy(self, x, y, edge_order=2):
        """Internal method used to obtain a Numpy gradient stencil model used to approximate the first derivative of a set of specific x- and y-coordinates. How edge cases are handled can be specified using the 'edge_order' keyword.

        Notes
        -----
        Uses forward or backward differences at the boundaries of a certain order specified by 'edge_order', and uses second-order accurate central differences in the interior points.

        Parameters
        ----------
        x: np.ndarray
            The x-coordinates from which the derivative model is constructed. These need not be equally spaced.
        y: np.ndarray
            The y-coordinates from which the derivative model is constructed. These need not be equally spaced.
        edge_order : int, optional
            Defines how the gradient is calculated at the boundaries: uses 'edge_order'-th order accurate differences. Limited to values 1 or 2, denoting first- or second-order accuracy; by default 2.

        Returns
        -------
        list[np.ndarray]
            The list containing the numpy gradient stencil models for the first derivative.
        str
            Information on the numpy gradient stencil models.
        """
        # retrieve list of numpy gradient stencil models
        results = [
            np.gradient(y, x, edge_order=edge_order)
            if (_ord_der == 1)
            else y
            if (_ord_der == 0)
            else None
            for _ord_der in range(self._order_derivative + 1)
        ]
        # return the stencil results + information string
        return (
            results,
            f'Numpy gradient stencil model for the first derivative (edge order cases handles with order of accuracy: {edge_order})',
        )

    #####################################################################
    # callable method that is used to perform numerical differentiation #
    #####################################################################
    def differentiate(self, x, y, **kwargs):
        """Method used to obtain the list of numerical derivates of the 'y'-variable in function of the 'x'-variable.

        Parameters
        ----------
        x : np.ndarray
            The 'x' variable.
        y : np.ndarray
            The 'y' variable.
        kwargs: dict
            Contains any keyword arguments necessary for the numerical differentiation. Which keyword arguments are used depends on the method:

            * unispline requires the 'degree_spline' and 'smoothing_factor_spline' keywords.
            * stencil_functions requires the 'my_order' and 'my_differencing_type' keywords.
            * gradient requires the 'edge_order' keyword.
            * fornberg requires the 'original_formulation' and 'interpolation_order' keywords.

            If no keyword arguments are passed, the default values are used, defined in each of the methods (in this class) responsible for the different differentiation types.

        Returns
        -------
        list[np.ndarray]
            The list of numerical derivates of the 'y'-variable in function of the 'x'-variable.
        str
            Information on the numerical derivative models.
        """
        # check if a linear interpolation should be done
        if self._perform_interpolation:
            # get the 1D interpolation
            _interpolator = interp1d(x, y)
            _x = np.linspace(
                x.min(),
                x.max(),
                num=((x.shape[0]) * self._interpolation_factor) - 1,
            )
            _y = _interpolator(_x)
        else:
            _x = x
            _y = y
        # run the numerical differentiation method using the selected kwargs or the default values (no kwargs supplied), and return the output
        # - use default values or use supplied kwarg values
        derivatives = (
            self._differentiation_executable(_x, _y)
            if (not kwargs)
            else self._differentiation_executable(_x, _y, **kwargs)
        )
        # check if sub-sampling needs to be done after interpolation
        if self._perform_interpolation:
            # perform the mapping from one to the other
            _map_idx = np.in1d(_x, x).nonzero()[0]
            # initialize output list
            der_sub_sample = []
            # perform sub-sampling
            for _my_der in derivatives[0]:
                if _my_der is None:
                    der_sub_sample.append(None)
                else:
                    der_sub_sample.append(_my_der[_map_idx])
            # return the derivatives
            return (der_sub_sample, derivatives[1])
        else:
            # return the derivatives
            return derivatives
