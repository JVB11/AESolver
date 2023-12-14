"""Python module that contains class that handles normalization of the Hough functions and their derivatives.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import logging
import sys
import numpy as np
from enum import Enum
from operator import attrgetter

# import custom integration module
from num_integrator import cni  # type: ignore


# set up the logger
logger = logging.getLogger(__name__)


# -------------------------------- HOUGH FUNCTION NORMALIZATION --------------------------------
# define the class that handles normalization of Hough functions and their derivatives
class HoughNormalizer(Enum):
    """Python class containing the relevant normalizations for the Hough functions (+ derivatives)."""

    # store the functions used to access the relevant normalizations for the Hough functions
    LEE2012 = attrgetter('_lee_2012_normalization_factor')
    PRAT2019 = attrgetter('_prat_2019_normalization_factor')

    # method that retrieves the requested normalization factor
    @classmethod
    def get_norm_factor(
        cls, norm_procedure, mu, hr, nr_omp_threads=4, use_parallel=False
    ):
        """Class method used to retrieve the normalization factor
        for the Hough functions.

        Parameters
        ----------
        norm_procedure : str
            A characteristic string that defines the (acceptable) normalization procedure. Options are: ['lee2012','prat2019'].
        mu : np.ndarray
            The corresponding cos(theta) == mu values for the values of the radial Hough functions.
        hr : np.ndarray
            The radial Hough function.
        nr_omp_threads : int, optional
            Number of OpenMP threads used for parallel integration, if 'use_parallel' is True.
        use_parallel : bool, optional
            If True, perform parallelized numerical integration, otherwise, perform serial numerical integration.

        Returns
        -------
        float
            The normalization factor.
        """
        # check if the requested normalization method is 'acceptable' for this code.
        if (npu := norm_procedure.upper()) in (
            acceptable_methods := cls._acceptable_normalizations()
        ):
            return getattr(cls, npu).value(cls)(
                mu, hr, nr_omp_threads, use_parallel
            )
        else:
            logger.error(
                f"The normalization procedure '{norm_procedure}' is unknown. Please provide an acceptable method (acceptable methods are: {acceptable_methods})."
            )
            sys.exit()

    # method that retrieves the acceptable normalization functions
    @classmethod
    def _acceptable_normalizations(cls):
        """Internal class method that retrieves the acceptable normalization procedures for this code.

        Returns
        -------
        list[str]
            Contains description of the acceptable normalization procedures.
        """
        # return the list of attributes stored in this enumeration class
        return [cls.__getitem__(_attr).name for _attr in cls.__members__]

    # method that computes the normalization factor for the Prat et al. (2019) normalization
    @classmethod
    def _prat_2019_normalization_factor(
        cls, mu, hr, nr_omp_threads=4, use_parallel=False
    ):
        """Internal class method that computes the normalization factor for the Prat et al. (2019) Hough function normalization.

        Notes
        -----
        The mu == cos(theta) argument is a dummy argument in this normalization convention.

        Parameters
        ----------
        mu : np.ndarray
            The corresponding cos(theta) == mu values for the values of the radial Hough functions.
        hr : np.ndarray
            The radial Hough function.
        nr_omp_threads : int, optional
            Number of OpenMP threads used for parallel integration, if 'use_parallel' is True.
        use_parallel : bool, optional
            If True, perform parallelized numerical integration, otherwise, perform serial numerical integration.

        Returns
        -------
        float
            The (multiplicative) normalizing factor.
        """
        # retrieve the maximum value of the radial Hough function,
        max_hr = np.abs(hr).max()
        # use that value to compute and return the normalizing (multiplicative) factor
        return 1.0 / max_hr

    # method that computes the normalization factor for the Lee (2012) normalization
    @classmethod
    def _lee_2012_normalization_factor(
        cls, mu, hr, nr_omp_threads=4, use_parallel=False
    ):
        """Internal class method that computes the normalization factor for the Lee (2012) Hough function normalization.

        Parameters
        ----------
        mu : np.ndarray
            The corresponding cos(theta) == mu values for the values of the radial Hough functions.
        hr : np.ndarray
            The radial Hough function.
        nr_omp_threads : int, optional
            Number of OpenMP threads used for parallel integration, if 'use_parallel' is True.
        use_parallel : bool, optional
            If True, perform parallelized numerical integration, otherwise, perform serial numerical integration.

        Returns
        -------
        float
            The (multiplicative) normalizing factor.
        """
        # compute the integrand for the normalization integral
        _integrand = hr**2.0
        # compute the normalization integral numerically
        _norm_integral = cni.integrate(
            'trapz', True, _integrand, mu, nr_omp_threads, use_parallel
        )
        # use that value to compute and return the normalizing (multiplicative) factor
        return 1.0 / np.sqrt(2.0 * np.pi * _norm_integral)
