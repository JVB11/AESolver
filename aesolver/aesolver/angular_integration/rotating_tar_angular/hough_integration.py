"""Python module that contains class that handles angular integration for rotating stars within the TAR.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# type checking done with strings:
from __future__ import annotations

# import statements
import logging
import sys
import numpy as np
from enum import Enum
from functools import partial
from scipy.interpolate import interp1d

# import custom module that computes numerical integrals TODO: check this!
from num_integrator import cni  # type: ignore

# type checking imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # type check types import
    from collections.abc import Callable
    from ...coupling_coefficients import QCCR


# set up the logger
logger = logging.getLogger(__name__)


# CONJUGATION FUNCTION DEFINITION + TYPING
def conjugator(m: int, conjugated: bool) -> int:
    """Performs complex conjugation of an azimuthal order, if necessary.

    Parameters
    ----------
    m : int
        Azimuthal order.
    conjugated : bool
        If True, change the sign of the azimuthal order, if False, leave it as it is.

    Returns
    -------
    int
        The potentially conjugated azimuthal order.
    """
    return -m if conjugated else m


# ------------------------- HOUGH FUNCTION INTEGRATION -------------------------
# define the class that handles the integration of the Hough functions and their derivatives
class HoughIntegration(Enum):
    """Enumeration class containing methods that handle the integration of combinations of Hough functions and their derivatives."""

    # define class attribute lambda function used to perform conjugation
    CONJER = partial(conjugator)

    # method that enforces the 'm-selection' rule
    @staticmethod
    def _phi_m_selection_rule(m_list: list[int]) -> bool:
        """Internal class method that implements the azimuthal 'm' selection rule for coupling coefficients in rotating stars.

        Parameters
        ----------
        m_list : list[int]
            The azimuthal orders of the modes.

        Returns
        -------
        bool
            True if the phi integration result is not zero (i.e. enforcing the m selection rule).
        """
        return sum(m_list) == 0

    # method that defines chebyshev-gauss quadrature for integrals of type 1: \int_{-1}^{1} (f(x)/sqrt(1-x^2)) dx, which is what we have for angular integrals that are divided by sin(x)!!!
    @staticmethod
    def _chebyshev_gauss_quad(f: 'Callable', order: int = 500) -> float | None:
        """Performs Chebyshev-Gauss quadrature on a provided function to evaluate an integral of the first type:

        .. math::
            \int_{-1}^{1} (\dfrac{f(x)}{sqrt(1-x^2)}) dx.

        Notes
        -----
        These are the type of integrals encountered when performing angular integration with integrands divided by sin(x)!!! TODO: this is an experimental feature. Should not be used extensively or without scrutiny.

        Parameters
        ----------
        f : typing.Callable
            The (interpolated) function that generates the integrand.
        order : int, optional
            The order of the Chebyshev-Gauss quadrature (i.e. how many internal points are used in the quadrature); by default 500.

        Returns
        -------
        float
            The result of the integration, as evaluated by Chebyshev-Gauss quadrature.
        """
        # compute the weight factor
        _weight = np.pi / float(order)
        # compute the internal points array
        _internal_points = (
            np.pi
            * ((2.0 * np.linspace(1.0, float(order), num=order)) - 1.0)
            / (2.0 * float(order))
        )
        # evaluate the function at each of the points
        try:
            return (
                _weight * f(np.cos(np.mod(_internal_points, 2.0 * np.pi)))
            ).sum(axis=0)
        except ValueError:
            return None

    # method used to compute the theta integral
    @classmethod
    def _compute_theta_integral(
        cls,
        my_obj: 'QCCR | object',
        descrips: list[str],
        div_by_sin: bool = False,
        mul_with_mu: bool = False,
        nr_omp_threads: int = 4,
        use_parallel: bool = False,
        num_integ_method: str = 'trapz',
        use_cheby_integration: bool = False,
        cheby_order_multiplier: int = 4,
        cheby_blow_up_protection: bool = False,
        cheby_blow_up_factor: float = 1e3,
    ) -> float:
        """Generic internal class method used to compute the theta angular integral.

        Parameters
        ----------
        my_obj : QCCR | object
            The specific object in which the Hough functions and their derivatives are stored.
        descrips : list[str]
            Contains specifications to retrieve the functions that make up the integrand.
        div_by_sin : bool, optional
            If True, divide the integrand by sin(theta) before integrating; by default False.
        mul_with_mu : bool, optional
            If True, multiply the integrand by cos(theta) before integrating; by default False.
        nr_omp_threads : int, optional
            Number of OpenMP threads used for parallel integration, if 'use_parallel' is True.
        use_parallel : bool, optional
            If True, perform parallelized numerical integration, otherwise, perform serial numerical integration.
        num_integ_method : str, optional
            The numerical integration method used to compute the integral; by default "trapz" (can also be "simpson").
        use_cheby_integration : bool, optional
            Instead of using the default integration method for numerical integration, use Chebyshev-Gauss quadrature along with linear interpolation; by default False.
        cheby_order_multiplier : int, optional
            Factor with which the size of the mu_array is multiplied to obtain the Chebyshev-Gauss quadrature order; by default 4.
        cheby_blow_up_protection : bool, optional
            Rely upon the regular integration method, except if the numerical integration result of that method exceeds 'cheby_blow_up_factor'; in that case, perform numerical integration using Chebyshev-Gauss quadrature; by default False.
        cheby_blow_up_factor : float, optional
            Blow-up determining factor used to determine when to use Chebyshev-Gauss quadrature if 'cheby_blow_up_protection' is True; by default 1e3.

        Returns
        -------
        float
            The theta integration result, or None if the integration failed.
        """
        # store the appropriate size of the internal utility lists
        _appropriate_size = len(descrips)
        # map the descriptions to the appropriate Hough functions/derivatives
        _first_term = getattr(my_obj, descrips[0])
        _mapped_array = np.empty(
            (_appropriate_size, _first_term.shape[0]), dtype=np.float64
        )
        _mapped_array[0, :] = _first_term
        for _i, _descr in enumerate(descrips[1:]):
            _mapped_array[_i + 1, :] = getattr(my_obj, _descr)
        # compute the integrand based on the array mapping
        _integrand = np.prod(_mapped_array, axis=0)
        # retrieve the mu-array
        _mu_arr = getattr(my_obj, '_mu_values')
        # check if the integrand needs a multiplication with mu = cos(theta)
        if mul_with_mu:
            _integrand *= _mu_arr
        # STRATEGY CAN DIFFER IF A DIVISIVE FACTOR IS USED AND A FLAG IS SET
        # - compute the integral and return the result
        if use_cheby_integration:
            # multiply integrand with sin(theta), if necessary, and create interpolating function
            if not div_by_sin:
                my_interpolator = interp1d(
                    _mu_arr, _integrand * np.sqrt(1.0 - (_mu_arr) ** 2.0)
                )
            else:
                my_interpolator = interp1d(_mu_arr, _integrand)
            # return the integration result, if the integration did not fail
            if (
                my_cheby_result := cls._chebyshev_gauss_quad(
                    my_interpolator,
                    order=cheby_order_multiplier * _mu_arr.shape[0],
                )
            ) is None:
                logger.error('Chebyshev integration failed. Now exiting.')
                sys.exit()
            else:
                return my_cheby_result
        else:
            # blow-up protection
            if cheby_blow_up_protection:
                # divide new integrand by sin(theta), if necessary
                if div_by_sin:
                    _new_integrand = _integrand / np.sqrt(
                        1.0 - (_mu_arr) ** 2.0
                    )
                else:
                    _new_integrand = _integrand
                # perform usual integration method
                regular_result = cni.integrate(
                    num_integ_method,
                    True,
                    _new_integrand,
                    _mu_arr,
                    nr_omp_threads,
                    use_parallel,
                )
                # check for blow-ups
                if regular_result > cheby_blow_up_factor:
                    # BLOW-UP: perform CG quadrature
                    if not div_by_sin:
                        my_interpolator = interp1d(
                            _mu_arr,
                            _integrand * np.sqrt(1.0 - (_mu_arr) ** 2.0),
                        )
                    else:
                        my_interpolator = interp1d(_mu_arr, _integrand)
                    # return the integration result
                    if (
                        my_cheby_result := cls._chebyshev_gauss_quad(
                            my_interpolator,
                            order=cheby_order_multiplier * _mu_arr.shape[0],
                        )
                    ) is not None:
                        return my_cheby_result
                    else:
                        logger.warning(
                            f'The regular result {regular_result} was used instead of the Chebyshev-Gauss result for {descrips}, because an extrapolation was required... (this was larger than {cheby_blow_up_factor})'
                        )
                        return regular_result
                else:
                    # return result
                    return regular_result
            # regular integration
            else:
                # divide integrand by sin(theta), if necessary
                if div_by_sin:
                    _integrand = _integrand / np.sqrt(1.0 - (_mu_arr) ** 2.0)
                # compute integral in mu-space
                return cni.integrate(
                    num_integ_method,
                    True,
                    _integrand,
                    _mu_arr,
                    nr_omp_threads,
                    use_parallel,
                )

    # method that adapts to complex conjugation of terms
    @classmethod
    def _adapt_to_complex_conjugation(
        cls, conj: bool | tuple[bool, bool, bool] | None, m_list: list[int]
    ) -> list[int]:
        """Internal class utility method that creates a list of azimuthal wavenumbers adapted to the criteria in the m selection rule.

        Parameters
        ----------
        conj : bool | tuple[bool, bool, bool] | None
            If None (default), no conjugation shall be performed for the m selection rule. If True, use complex conjugate of first mode. If list[bool], use the corresponding boolean values to decide which term is complex conjugated.
        m_list : list[int]
            The list of azimuthal wavenumbers of the modes.

        Returns
        -------
        list[int]
            The list of adapted azimuthal wavenumbers of the modes.
        """
        # adjust for conjugation in phi integration
        if conj is None:
            # - no conjugation
            return m_list
        elif isinstance(conj, tuple):
            # check length of tuple
            if len(conj) != 3:
                logger.error(
                    f"Length of supplied 'conj' tuple ({conj}) is not 3. Cannot use this information. Now exiting."
                )
                sys.exit()
            # - adjust for conjugation
            return list(map(cls.CONJER.value, m_list, conj))
        elif isinstance(conj, bool):
            # - adjust for conjugation in azimuthal order of first mode
            m_phi_list = [cls.CONJER.value(m_list[0], conj)]
            m_phi_list.extend(m_list[1:])
            return m_phi_list
        else:
            logger.error("Cannot recognize the type of 'conj' in '_adapt_to_complex_conjugation', now exiting.")
            sys.exit()

    # method that handles the requested numerical integration over the Hough functions
    @classmethod
    def quadratic_angular_integral(
        cls,
        my_obj: 'QCCR | object',
        descrips: list[str],
        m_list_for_azimuthal_check: list[int] | None = None,
        only_theta: bool=False,
        conj: bool | tuple[bool, bool, bool] | None = None,
        div_by_sin: bool = False,
        mul_with_mu: bool = False,
        nr_omp_threads: int = 4,
        use_parallel: bool = False,
        num_integ_method: str = 'trapz',
        use_cheby_integration: bool = False,
        cheby_order_multiplier: int = 4,
        cheby_blow_up_protection: bool = False,
        cheby_blow_up_factor: float = 1e3,
    ) -> float:
        """Class method that computes angular integrals for the terms specific to the computations necessary for computing coupling coefficients in rotating stars numerically.

        Parameters
        ----------
        my_obj : QCCR | object
            The specific object in which the Hough functions and their derivatives are stored.
        descrips : list[str]
            Contains specifications to retrieve the functions that make up the integrand.
        conj : bool | tuple[bool, bool, bool] | None, optional
            If None (default), no conjugation shall be performed for the m selection rule. If True, use complex conjugate of first mode. If list[bool], use the corresponding boolean values to decide which term is complex conjugated; by default None.
        div_by_sin : bool, optional
            If True, divide the integrand by sin(theta) before integrating; by default False.
        mul_with_mu : bool, optional
            If True, multiply the integrand by cos(theta) before integrating; by default False.
        m_list_for_azimuthal_check : list[int], optional
            If None, do not check the azimuthal selection rule. If list[int], this is the list of azimuthal wavenumbers of the modes that will be used to check the fulfillment of the azimuthal selection rule; by default None.
        only_theta: bool, optional
            If True, we do not care about the azimuthal selection rule, and compute the theta part of the angular integral, despite the rule not being fulfilled. If False, we care about the selection rule; by default False.
        num_integ_method : str, optional
            The numerical integration method used to compute the integral; by default "trapz" (can also be "simpson").
        use_cheby_integration : bool, optional
            Instead of using the default integration method for numerical integration, use Chebyshev-Gauss quadrature along with linear interpolation; by default False.
        cheby_order_multiplier : int, optional
            Factor with which the size of the mu_array is multiplied to obtain the Chebyshev-Gauss quadrature order; by default 4.
        cheby_blow_up_protection : bool, optional
            Rely upon the regular integration method, except if the numerical integration result of that method exceeds 'cheby_blow_up_factor'; in that case, perform numerical integration using Chebyshev-Gauss quadrature; by default False.
        cheby_blow_up_factor : float, optional
            Blow-up determining factor used to determine when to use Chebyshev-Gauss quadrature if 'cheby_blow_up_protection' is True; by default 1e3.

        Returns
        -------
        float
            The result of the angular integration.
        """
        # set the pre-check boolean
        if only_theta or (m_list_for_azimuthal_check is None):
            _pre_check = True  # no check needed
        else:
            # use dummy list '_adapt' to check the azimuthal selection rule
            _adapt = cls._adapt_to_complex_conjugation(conj=conj, m_list=m_list_for_azimuthal_check)
            _pre_check = cls._phi_m_selection_rule(_adapt)
        # if pre-check fulfilled, compute the integral, otherwise, return zero
        if _pre_check:
            # compute the integral over the theta coordinate
            theta_integral = cls._compute_theta_integral(
                my_obj=my_obj,
                descrips=descrips,
                div_by_sin=div_by_sin,
                mul_with_mu=mul_with_mu,
                nr_omp_threads=nr_omp_threads,
                use_parallel=use_parallel,
                num_integ_method=num_integ_method,
                use_cheby_integration=use_cheby_integration,
                cheby_blow_up_factor=cheby_blow_up_factor,
                cheby_blow_up_protection=cheby_blow_up_protection,
                cheby_order_multiplier=cheby_order_multiplier,
            )
            # multiply with the phi coordinate integration result
            return 2.0 * np.pi * theta_integral
        else:
            return 0.0
