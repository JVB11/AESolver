"""Python file containing the superclass that defines functionalities for the computation of coupling coefficients.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import logging
import numpy as np

# import intra-package module
from .super_coupling_coefficients import SuperCouplingCoefficient

# import debugging module enumeration
from ..enumeration_files.enumerated_debug_properties import (
    DebugMinMaxCouplingCoefficientRotating as DmmCCr,
)

# import custom module
from num_deriv import NumericalDifferentiator

# type checking imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # import frequency_handler object
    from ...frequency_handling import FH

    # type checking types
    from typing import Any


# set up logger
logger = logging.getLogger(__name__)


# ignore numpy warnings
np.seterr('ignore')


# ------------------ COUPLING SUPERCLASS (Lee, 2012) -------------------
class CouplingCoefficientRotating(SuperCouplingCoefficient):
    """Python superclass that implements the generic method used to compute mode coupling coefficients in rotating stars, based on Lee (2012).

    Parameters
    ----------
    nr_modes : int
        The number of modes involved in the computation of the coupling coefficient.
    kwargs_diff_div : dict or None, optional
        The keyword arguments dictionary that sets arguments to be used for computing divergences and their derivatives. If None, no keyword arguments are passed; by default None.
    kwargs_diff_terms : dict or None, optional
        The keyword arguments dictionary that sets arguments to be used for computing derivatives of terms in the coupling coefficient. If None, no keyword arguments are passed; by default None.
    diff_terms_method : str, optional
        Denotes the numerical method used to compute (numerical) derivatives for the terms in the coupling coefficient, if necessary; by default 'gradient'.
    polytropic : bool, optional
        If True, perform the necessary actions to compute the coupling coefficient for polytrope input. If False, assume stellar evolution model input for the computations; by default False.
    store_debug : bool, optional
        If True, store debug properties/attributes. If False, do not store these attributes; by default False.
    """

    # attribute type declarations
    _rot_ang_freq: 'float | complex |np.ndarray'
    # mode-specific attributes
    _z_1_mode_1: 'np.ndarray'
    _z_2_mode_1: 'np.ndarray'
    _x_z_1_mode_1: 'np.ndarray'
    _z_2_over_c_1_omega_mode_1: 'np.ndarray'
    _x_z_2_over_c_1_omega_mode_1: 'np.ndarray'
    _rad_der_x_z_1_mode_1: 'np.ndarray'
    _rad_der_x_z_2_c_1_omega_mode_1: 'np.ndarray'
    _rad_diverg_mode_1: 'np.ndarray'
    _z_1_mode_2: 'np.ndarray'
    _z_2_mode_2: 'np.ndarray'
    _x_z_1_mode_2: 'np.ndarray'
    _z_2_over_c_1_omega_mode_2: 'np.ndarray'
    _x_z_2_over_c_1_omega_mode_2: 'np.ndarray'
    _rad_der_x_z_1_mode_2: 'np.ndarray'
    _rad_der_x_z_2_c_1_omega_mode_2: 'np.ndarray'
    _rad_diverg_mode_2: 'np.ndarray'
    _z_1_mode_3: 'np.ndarray'
    _z_2_mode_3: 'np.ndarray'
    _x_z_1_mode_3: 'np.ndarray'
    _z_2_over_c_1_omega_mode_3: 'np.ndarray'
    _x_z_2_over_c_1_omega_mode_3: 'np.ndarray'
    _rad_der_x_z_1_mode_3: 'np.ndarray'
    _rad_der_x_z_2_c_1_omega_mode_3: 'np.ndarray'
    _rad_diverg_mode_3: 'np.ndarray'
    # mode specific attributes for debugging purposes
    _z_1_mode_1_max_min: 'np.ndarray'
    _z_2_mode_1_max_min: 'np.ndarray'
    _x_z_1_mode_1_max_min: 'np.ndarray'
    _z_2_over_c_1_omega_mode_1_max_min: 'np.ndarray'
    _x_z_2_over_c_1_omega_mode_1_max_min: 'np.ndarray'
    _rad_der_x_z_1_mode_1_max_min: 'np.ndarray'
    _rad_der_x_z_2_c_1_omega_mode_1_max_min: 'np.ndarray'
    _rad_diverg_mode_1_max_min: 'np.ndarray'
    _z_1_mode_2_max_min: 'np.ndarray'
    _z_2_mode_2_max_min: 'np.ndarray'
    _x_z_1_mode_2_max_min: 'np.ndarray'
    _z_2_over_c_1_omega_mode_2_max_min: 'np.ndarray'
    _x_z_2_over_c_1_omega_mode_2_max_min: 'np.ndarray'
    _rad_der_x_z_1_mode_2_max_min: 'np.ndarray'
    _rad_der_x_z_2_c_1_omega_mode_2_max_min: 'np.ndarray'
    _rad_diverg_mode_2_max_min: 'np.ndarray'
    _z_1_mode_3_max_min: 'np.ndarray'
    _z_2_mode_3_max_min: 'np.ndarray'
    _x_z_1_mode_3_max_min: 'np.ndarray'
    _z_2_over_c_1_omega_mode_3_max_min: 'np.ndarray'
    _x_z_2_over_c_1_omega_mode_3_max_min: 'np.ndarray'
    _rad_der_x_z_1_mode_3_max_min: 'np.ndarray'
    _rad_der_x_z_2_c_1_omega_mode_3_max_min: 'np.ndarray'
    _rad_diverg_mode_3_max_min: 'np.ndarray'
    _l_list: list[int]
    _M_star: float
    _freq_handler: 'FH'

    # superclass entity initialization
    def __init__(
        self,
        nr_modes: int,
        kwargs_diff_div: 'dict[str, Any] | None' = None,
        kwargs_diff_terms: 'dict[str, Any] | None' = None,
        diff_terms_method: str = 'gradient',
        polytropic: bool = False,
        analytic_polytrope: bool = False,
        store_debug: bool = False,
    ) -> None:
        # initialize the superclass for coupling coefficients
        super().__init__(
            nr_modes=nr_modes,
            kwargs_diff_div=kwargs_diff_div,
            kwargs_diff_terms=kwargs_diff_terms,
            diff_terms_method=diff_terms_method,
            polytropic=polytropic,
            store_debug=store_debug,
            analytic_polytrope=analytic_polytrope,
        )

    # GETTER/SETTER METHODS

    # INITIALIZATION COMPUTATION METHODS

    # method that computes and stores the radial vectors necessary for mode coupling calculations
    def _compute_radial_vectors(self) -> None:
        """Internal method that computes the radial vectors necessary for mode coupling calculations."""
        # loop over the numbers of the modes involved in the computations, to store the dimensionless radial vectors
        for _i in range(1, self._nr_modes + 1):
            # compute and store the z1 and z2 functions of Lee (2012)
            # NOTE: this only holds within the Cowling approximation!!!!!!!!
            setattr(
                self,
                f'_z_1_mode_{_i}',
                getattr(self, f'_y_1_mode_{_i}')
                * self._x ** (self._l_list[_i - 1] - 2)
                * np.sqrt(4.0 * np.pi),
            )
            setattr(
                self,
                f'_z_2_mode_{_i}',
                getattr(self, f'_y_2_mode_{_i}')
                * self._x ** (self._l_list[_i - 1] - 2)
                * np.sqrt(4.0 * np.pi),
            )
            # compute additional products of radial vectors
            setattr(
                self,
                f'_x_z_1_mode_{_i}',
                self._x * getattr(self, f'_z_1_mode_{_i}'),
            )
            setattr(
                self,
                f'_z_2_over_c_1_omega_mode_{_i}',
                getattr(self, f'_z_2_mode_{_i}')
                / (
                    self._c_1
                    * self._freq_handler.dimless_corot_omegas[_i - 1] ** (2.0)
                ),
            )
            setattr(
                self,
                f'_x_z_2_over_c_1_omega_mode_{_i}',
                self._x * getattr(self, f'_z_2_over_c_1_omega_mode_{_i}'),
            )
            # add the debug attributes, if necessary
            if self._store_debug:
                self._add_debug_attribute(f'_z_1_mode_{_i}')
                self._add_debug_attribute(f'_z_2_mode_{_i}')
                self._add_debug_attribute(f'_x_z_1_mode_{_i}')
                self._add_debug_attribute(f'_z_2_over_c_1_omega_mode_{_i}')
                self._add_debug_attribute(f'_x_z_2_over_c_1_omega_mode_{_i}')

    # method that computes and stores radial gradients for the radial vectors
    def _compute_radial_gradients(self, diff_method: str) -> None:
        """Internal method that computes the radial gradients for the radial vectors necessary for the quadratic mode coupling computations in Lee(2012), based on the more generic GYRE definitions of the eigenfunctions.

        Parameters
        ----------
        diff_method : str
            String that specifier which numerical differentiation method should be used in the NumericalDifferentiator object.
        """
        # initialize the numerical differentiation object suited for computing the first-order numerical derivatives needed to compute the divergences of the linear eigenfunctions
        _grad = NumericalDifferentiator(
            order_derivative=1, differentiation_method=diff_method
        )
        # use dummy variable for rho: adjust for polytrope data
        if self._use_polytropic:
            _rho = self._rho_profile
        else:
            _rho = self._rho
        # divert actions according to whether numerical or symbolical gradients are computed
        # -- SYMBOLIC
        if self._use_symbolic_derivative:
            # obtain necessary structure coefficient combinations
            _V_over_G1 = (self._V_2 * (self._x**2.0)) / self._Gamma_1
            # loop over the numbers of the modes involved in the computations, to store the dimensionless radial gradients
            for _i in range(1, self._nr_modes + 1):
                # compute the corotating frame dimensionless frequency factor
                _omega_dimless_sq = self._freq_handler.dimless_corot_omegas[
                    _i - 1
                ] ** (2.0)
                _c1_omega = self._c_1 * _omega_dimless_sq
                # compute the radial derivatives of the x*z1 function based on definition A1 in Lee(2012):
                # r*dz1/dr = ((V/Gamma_1)-3)*z1 + ((lambda/(c1*omega_bar^2))-(V/Gamma_1))*z2
                # --> d(x*z1)/dx = d(r*z1)/dr = z1 + r*dz1/dr = ((V/Gamma_1)-2)*z1 + ((lambda/(c1*omega_bar^2))-(V/Gamma_1))*z2
                _prefactor1 = _V_over_G1 - 2.0
                _prefactor2 = (
                    getattr(self, f'_lambda_mode_{_i}')[0] / _c1_omega
                ) - _V_over_G1
                _rad_der_y1_list = _prefactor1 * getattr(
                    self, f'_z_1_mode_{_i}'
                ) + _prefactor2 * getattr(self, f'_z_2_mode_{_i}')
                setattr(self, f'_rad_der_x_z_1_mode_{_i}', _rad_der_y1_list)
                # compute the radial derivatives of the x*z2/(c1*omega_bar^2) function based on definition A2 in Lee(2012):
                # r*dz2/dr = x*dz2/dx = (c1*omega_bar^2-As)*z1 + (1-U+As)*z2
                # d/dx(a*x*z2) = x*a*dz2/dx + a*z2 + x*z2*da/dx
                # NOTE: c1(r) = (r^3/R^3)*(M/Mr(r)) = x^3 * (M/Mr(r))
                # NOTE: d(c1(r))/dr = (r^2/R^3)(M/Mr(r))[3 - (4*Pi*rho*r^3/Mr(r))]
                # --> d(x*z2/(c1*omega_bar^2))/dx = (1-(As/(c1*omega_bar^2)))*z1 + ((2-U+As)/(c1*omega_bar^2))*z2 + (z2*R^3/(omega_bar^2*M))*[(4*Pi*rho)-(3*Mr/r^3)]
                # ---> IMPLEMENTATION FORM:
                # ---> d(x*z2/(c1*omega_bar^2))/dx = (1-(As/(c1*omega_bar^2)))*z1 - ((1+U-As)/(c1*omega_bar^2))*z2 + z2*(4*Pi*rho)/(<rho>*omega_bar^2)
                # ---> WITH <rho> = M / R^3
                _avg_rho = self._M_star / self._R_star ** (3.0)
                _prefactor3 = 1.0 - (self._As / _c1_omega)
                _prefactor4 = (self._As - self._U - 1.0) / _c1_omega
                _prefactor5 = (4.0 * np.pi * _rho) / (
                    _avg_rho * _omega_dimless_sq
                )
                _rad_der_y2_list = _prefactor3 * getattr(
                    self, f'_z_1_mode_{_i}'
                ) + (_prefactor4 + _prefactor5) * getattr(
                    self, f'_z_2_mode_{_i}'
                )
                setattr(
                    self,
                    f'_rad_der_x_z_2_c_1_omega_mode_{_i}',
                    _rad_der_y2_list,
                )
        # -- NUMERICAL
        else:
            # loop over the numbers of the modes involved in the computations, to store the dimensionless radial gradients
            for _i in range(1, self._nr_modes + 1):
                # compute the radial derivatives of the x*z1 function using numerical derivative expressions
                _, _rad_der_y1_list = self._compute_der(
                    differentiator=_grad,
                    x=self._x,
                    y=getattr(self, f'_x_z_1_mode_{_i}'),
                    kwargs_diff=self._kwargs_diff_div,
                )
                setattr(self, f'_rad_der_x_z_1_mode_{_i}', _rad_der_y1_list)
                # compute the radial derivatives of the x*z2/(c1*omega_bar^2) function
                _, _rad_der_y2_list = self._compute_der(
                    differentiator=_grad,
                    x=self._x,
                    y=getattr(self, f'_x_z_2_over_c_1_omega_mode_{_i}'),
                    kwargs_diff=self._kwargs_diff_div,
                )
                setattr(
                    self,
                    f'_rad_der_x_z_2_c_1_omega_mode_{_i}',
                    _rad_der_y2_list,
                )
                # (1.0 / self._x) * _rad_der_y2_list)
        # add debug attributes, if necessary
        if self._store_debug:
            for _i in range(1, self._nr_modes + 1):
                self._add_debug_attribute(f'_rad_der_x_z_1_mode_{_i}')
                self._add_debug_attribute(f'_rad_der_x_z_2_c_1_omega_mode_{_i}')

    # method that computes the radial part of the divergence factors
    def _compute_divergence_radial_part(self) -> None:
        """Internal method that computes and sets the radial part of the divergences of the mode eigenfunctions within the TAR."""
        # compute V / Gamma_1 from the stored variables
        _V_G1 = self._V_2 * (self._x**2.0) / self._Gamma_1
        # loop over the numbers of the modes involved in the computations, to store compute radial parts of divergences
        for _i in range(1, self._nr_modes + 1):
            # get the z-difference
            _z_diff = getattr(self, f'_z_1_mode_{_i}') - getattr(
                self, f'_z_2_mode_{_i}'
            )
            # get and store the radial part of the divergence
            setattr(self, f'_rad_diverg_mode_{_i}', _z_diff * _V_G1)
            # add debug attributes, if necessary
            if self._store_debug:
                self._add_debug_attribute(f'_rad_diverg_mode_{_i}')

    # method that fills up the debug attributes for specific modes
    def _fill_specific_mode_debug_attributes(self) -> None:
        """Fills up the specific debug attributes for the modes."""
        if self._store_debug:
            for _i in range(1, self._nr_modes + 1):
                self._fill_debug_attributes(
                    DmmCCr.get_mode_specific_attrs(mode_nr=_i)
                )
