"""Python module containing class that computes and stores the necessary additional information for polytropic structure model files.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import logging
import numpy as np

# import intra-package conversion information
from .enumeration_files import PdC
from .enumeration_files import PSo

# type checking imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # import type checking types
    from collections.abc import Callable
    from typing import Any, Self

    # import numpy typing
    import numpy.typing as npt


# set up the logger
logger = logging.getLogger(__name__)


# define the class that computes additional structure info
class PolytropicAdditionalStructureInformation:
    """Contains/computes and stores additional polytropic structure model information."""

    # attribute type declarations
    _structure_dict: 'dict[str, Any]'
    _n: int
    _z: 'npt.NDArray[np.float64]'
    _z_s: float
    _theta: 'npt.NDArray[np.float64]'
    _dtheta: 'npt.NDArray[np.float64]'
    _gamma1: 'npt.NDArray[np.float64]'
    _G: float
    _mass: float
    _radius: float
    _aux_mass: 'npt.NDArray[np.float64]'
    _aux_s: float
    _avg_rho: float
    _x: 'npt.NDArray[np.float64]'
    _rho0: float
    _p0: float
    _alpha_conversion: float
    _dtheta2: 'npt.NDArray[np.float64]'
    _dtheta3: 'npt.NDArray[np.float64]'
    _rho_norm: 'npt.NDArray[np.float64]'
    _drho_dz: 'npt.NDArray[np.float64]'
    _drho2_dz2: 'npt.NDArray[np.float64]'
    _drho3_dz3: 'npt.NDArray[np.float64]'
    _p_norm: 'npt.NDArray[np.float64]'
    _dp_dz: 'npt.NDArray[np.float64]'
    _dp2_dz2: 'npt.NDArray[np.float64]'
    _dp3_dz3: 'npt.NDArray[np.float64]'
    _dgrav_pot_dz: 'npt.NDArray[np.float64]'
    _dgrav_pot2_dz2: 'npt.NDArray[np.float64]'
    _dgrav_pot3_dz3: 'npt.NDArray[np.float64]'
    _drho: 'npt.NDArray[np.float64]'
    _drho2: 'npt.NDArray[np.float64]'
    _drho3: 'npt.NDArray[np.float64]'
    _dp: 'npt.NDArray[np.float64]'
    _dp2: 'npt.NDArray[np.float64]'
    _dp3: 'npt.NDArray[np.float64]'
    _dgrav_pot: 'npt.NDArray[np.float64]'
    _dgrav_pot2: 'npt.NDArray[np.float64]'
    _dgrav_pot3: 'npt.NDArray[np.float64]'
    _r: 'npt.NDArray[np.float64]'
    _p: 'npt.NDArray[np.float64]'
    _rho: 'npt.NDArray[np.float64]'
    _dt_ratio: 'npt.NDArray[np.float64]'
    _dt_t_ratio: 'npt.NDArray[np.float64]'
    _v_2: 'npt.NDArray[np.float64]'
    _v: 'npt.NDArray[np.float64]'
    _u: 'npt.NDArray[np.float64]'
    _c1: 'npt.NDArray[np.float64]'
    _as: 'npt.NDArray[np.float64]'
    _mr: 'npt.NDArray[np.float64]'
    _surf_grav: 'npt.NDArray[np.float64]'
    _brunt_squared: 'npt.NDArray[np.float64]'

    def __init__(
        self,
        model_info_dictionary_list: 'list[dict[str, Any]]',
        model_mass: float,
        model_radius: float,
    ) -> None:
        # store the structure model information dict
        self._structure_dict = model_info_dictionary_list[0]
        # extract information and store it in convenience variables
        self._n = self._structure_dict['n_poly']
        self._z = self._structure_dict['z']
        self._z_s = self._z[-1]
        self._theta = self._structure_dict['theta']
        self._dtheta = self._structure_dict['dtheta']
        self._gamma1 = self._structure_dict['Gamma_1']
        self._G = self._structure_dict['G']
        # store mass and radius of the model
        self._mass = model_mass
        self._radius = model_radius
        # compute and store the auxiliary mass variable
        self._store_auxiliary_mass()
        # compute and store the average density
        self._store_average_density()
        # compute and store the fractional radius
        self._store_fractional_radius()
        # compute the rho and P zero factors
        self._compute_rho_zero()
        self._compute_p_zero()
        # compute the alpha conversion factor
        self._compute_alpha_conversion_factor()
        # compute and store the additional structure info
        self._compute_additional_structure_info()
        # convert z derivatives into their r derivative counterparts
        self._convert_z_derivatives()
        # compute dimensioned profiles
        self._compute_dimensioned_profiles()
        # compute structure factors
        self._compute_structure_factors()
        # compute buoyancy frequency
        self._compute_buoyancy_freq()

    @property
    def property_dict(self) -> 'dict[str, Any]':
        """Returns the dictionary containing the additional requested properties for the polytrope structure files.

        Returns
        -------
        dict[str, Any]
            Contains the requested properties.
        """
        return {
            _k: _f(self)
            for _k, _f in PSo.get_output_creation_info_dict().items()
        }

    def _store_auxiliary_mass(self) -> None:
        """Computes and stores the auxiliary mass variable (typically called mu)."""
        # store auxiliary mass variable
        self._aux_mass = -1.0 * (self._z**2.0) * self._dtheta
        # store its value at the surface
        self._aux_s = self._aux_mass[-1]

    def _store_average_density(self) -> None:
        """Computes and stores the average mass density."""
        self._avg_rho = (3.0 / (4.0 * np.pi)) * (
            self._mass / (self._radius**3.0)
        )

    def _store_fractional_radius(self) -> None:
        """Computes and store the fractional radius."""
        self._x = self._z / self._z_s

    def _compute_rho_zero(self) -> None:
        """Computes the rho_0 factor."""
        self._rho0 = (self._avg_rho / 3.0) * ((self._z_s**3.0) / self._aux_s)

    def _compute_p_zero(self) -> None:
        """Computes the P_0 factor."""
        # compute scaling factors
        _rat = self._rho0 / self._avg_rho
        _nn = ((self._z_s) ** (1.0 - (3.0 / self._n)) / (self._n + 1.0)) * (
            (4.0 * np.pi) ** (1.0 / self._n)
            * self._aux_s ** ((1.0 / self._n) - 1.0)
        )
        _wn = _nn * ((3.0 / (4.0 * np.pi)) * _rat) ** (1.0 + (1.0 / self._n))
        # compute the p0 factor
        try:
            self._p0 = (
                _wn * (self._G * (self._mass**2.0)) / (self._radius**4.0)
            )[0]
        except IndexError:
            self._p0 = _wn * (self._G * (self._mass**2.0)) / (self._radius**4.0)

    def _compute_alpha_conversion_factor(self) -> None:
        """Computes and stores the alpha conversion factor used to convert between z and r."""
        self._alpha_conversion = np.sqrt(
            (self._p0 * (1.0 + self._n))
            / (self._G * 4.0 * np.pi * (self._rho0**2.0))
        )

    # method that computes several structure information arrays
    def _compute_additional_structure_info(self) -> None:
        """Computes several structure information arrays from polytropic structure file data."""
        # compute and store the theta derivatives
        self._compute_store_theta_derivatives()
        # compute and store mass density and derivatives
        self._compute_rho()
        self._compute_rho_z_derivatives()
        # compute and store the pressure profile and derivatives
        self._compute_p()
        self._compute_p_z_derivatives()
        # compute the gravitational potential derivatives
        self._compute_grav_pot_derivatives()

    def _compute_store_theta_derivatives(self) -> None:
        """Compute polytropic theta parameter derivatives via an analytical formula."""
        # compute and store second derivative
        self._compute_dtheta_2()
        # compute and store third derivative
        self._compute_dtheta_3()

    def _compute_dtheta_2(self) -> None:
        """Compute and store the second polytropic theta parameter z-derivative."""
        self._dtheta2 = np.nan_to_num(
            (-(self._theta**self._n)) - (2.0 / self._z) * self._dtheta
        )

    def _compute_dtheta_3(self) -> None:
        """Compute and store the third polytropic theta parameter z-derivative."""
        self._dtheta3 = np.nan_to_num(
            ((2.0 / self._z) * self._theta**self._n)
            + (-self._n * self._theta ** (self._n - 1.0) + (6.0 / self._z**2.0))
            * self._dtheta
        )

    def _compute_rho(self) -> None:
        """Computes and stores the normalized mass density profile."""
        self._rho_norm = self._theta ** (self._n)

    def _compute_rho_z_derivatives(self) -> None:
        """Computes and stores the normalized mass density z-derivatives."""
        # store convenience variables
        _n_1 = self._n - 1.0
        _n_2 = _n_1 - 1.0
        _n_3 = _n_2 - 1.0
        # first z-derivative
        self._compute_drho_dz(_n_1)
        # second z-derivative
        self._compute_drho_dz_2(_n_1, _n_2)
        # third z-derivative
        self._compute_drho_dz_3(_n_1, _n_2, _n_3)

    def _compute_drho_dz(self, n_1: float | int) -> None:
        """Computes the first normalized mass density z-derivative.

        Parameters
        ----------
        n_1 : float | int
            Float representation of polytropic index minus 1.
        """
        self._drho_dz = self._n * self._dtheta * (self._theta**n_1)

    def _compute_drho_dz_2(self, n_1: float | int, n_2: float | int) -> None:
        """Computes the second normalized mass density z-derivative.

        Parameters
        ----------
        n_1 : float | int
            Float representation of polytropic index minus 1.
        n_2 : float | int
            Float representation of polytropic index minus 2.
        """
        self._drho2_dz2 = self._n * (
            (n_1 * self._theta**n_2 * self._dtheta**2.0)
            + self._theta**n_1 * self._dtheta2
        )

    def _compute_drho_dz_3(
        self, n_1: float | int, n_2: float | int, n_3: float | int
    ) -> None:
        """Computes the third normalized mass density z-derivative.

        Parameters
        ----------
        n_1 : float | int
            Float representation of polytropic index minus 1.
        n_2 : float | int
            Float representation of polytropic index minus 2.
        n_3 : float | int
            Float representation of polytropic index minus 3.
        """
        self._drho3_dz3 = self._n * (
            n_1
            * (
                (n_2 * self._theta**n_3 * self._dtheta**3.0)
                + (3.0 * self._theta**n_2 * self._dtheta2 * self._dtheta)
            )
            + self._theta**n_1 * self._dtheta3
        )

    def _compute_p(self):
        """Computes and stores the normalized pressure profile."""
        self._p_norm = self._theta ** (self._n + 1.0)

    def _compute_p_z_derivatives(self):
        """Computes and stores the normalized pressure profile z-derivatives."""
        # store convenience variables
        _np = self._n + 1.0
        _n_1 = self._n - 1.0
        _n_2 = _n_1 - 1.0
        # first z-derivative
        self._compute_dp_dz(_np)
        # second z-derivative
        self._compute_dp_dz_2(_np, _n_1)
        # third z-derivative
        self._compute_dp_dz_3(_np, _n_1, _n_2)

    def _compute_dp_dz(self, n_1: float | int) -> None:
        """Computes the first normalized pressure profile z-derivative.

        Parameters
        ----------
        n_1 : float | int
            Float representation of polytropic index plus 1.
        """
        self._dp_dz = n_1 * self._theta**self._n * self._dtheta

    def _compute_dp_dz_2(self, n_1: float | int, n_2: float | int):
        """Computes the second normalized pressure profile z-derivative.

        Parameters
        ----------
        n_1 : float | int
            Float representation of polytropic index plus 1.
        n_2 : float | int
            Float representation of polytropic index minus 1.
        """
        self._dp2_dz2 = (
            n_1
            * self._theta**n_2
            * (self._n * self._dtheta**2.0 + self._theta * self._dtheta2)
        )

    def _compute_dp_dz_3(
        self, n_1: float | int, n_2: float | int, n_3: float | int
    ) -> None:
        """Computes the third normalized pressure profile z-derivative.

        Parameters
        ----------
        n_1 : float | int
            Float representation of polytropic index plus 1.
        n_2 : float | int
            Float representation of polytropic index minus 1.
        n_3 : float | int
            Float representation of polytropic index minus 2.
        """
        self._dp3_dz3 = (
            n_1
            * self._theta**n_3
            * (
                n_2 * self._n * self._dtheta**3.0
                + (3.0 * self._n * self._theta * self._dtheta * self._dtheta2)
                + self._theta**2.0 * self._dtheta3
            )
        )

    def _compute_grav_pot_derivatives(self):
        """Computes the normalized gravitational potential and its z-derivatives."""
        # store convenience variable
        _np = self._n + 1.0
        # first z-derivative
        self._compute_d_grav_pot(_np)
        # second z-derivative
        self._compute_d_grav_pot_2(_np)
        # third z-derivative
        self._compute_d_grav_pot_3(_np)

    def _compute_d_grav_pot(self, n_1: float | int) -> None:
        """Compute the first gravitational potential z-derivative.

        Parameters
        ----------
        n_1 : float | int
            Float representation of polytropic index plus 1.
        """
        self._dgrav_pot_dz = -n_1 * self._dtheta

    def _compute_d_grav_pot_2(self, n_1: float | int) -> None:
        """Compute the second gravitational potential z-derivative.

        Parameters
        ----------
        n_1 : float | int
            Float representation of polytropic index plus 1.
        """
        self._dgrav_pot2_dz2 = -n_1 * self._dtheta2

    def _compute_d_grav_pot_3(self, n_1: float | int) -> None:
        """Compute the third gravitational potential z-derivative.

        Parameters
        ----------
        n_1 : float | int
            Float representation of polytropic index plus 1.
        """
        self._dgrav_pot3_dz3 = -n_1 * self._dtheta3

    def _convert_z_derivatives(self) -> None:
        """Convert the z-derivatives into their r-derivative counterparts."""
        # obtain information on the necessary conversion
        _conv_info = PdC.get_der_conversion_information()
        # perform conversions to r derivatives
        for _z_d, _n, _o, _c in _conv_info:
            self._convert_to_r_derivative(
                z_der=_z_d, r_der_name=_n, order=_o, conversion_call=_c
            )

    def _convert_to_r_derivative(
        self,
        z_der: str,
        r_der_name: str,
        order: int,
        conversion_call: 'Callable[[Self], float]',
    ) -> None:
        """Converts a given z-derivative into its r-derivative counterpart.

        Parameters
        ----------
        z_der : str
            Describes which z-derivative is to be converted.
        r_der_name : str
            The name under which the r-derivative will be stored.
        order : int
            The order of the derivative.
        conversion_call : Callable[[Self], float]
            Retrieves the multiplication factor for conversion.
        """
        setattr(
            self,
            r_der_name,
            (
                self._alpha_conversion ** (-order)
                * getattr(self, z_der)
                * conversion_call(self)
            ),
        )

    def _compute_dimensioned_profiles(self) -> None:
        """Computes profiles with dimensions from the dimensionless attributes of a polytrope structure file."""
        # radius profile
        self._compute_radius()
        # pressure profile
        self._compute_pressure()
        # mass density profile
        self._compute_density()

    def _compute_radius(self) -> None:
        """Compute and store the radius profile."""
        self._r = self._z * self._alpha_conversion

    def _compute_pressure(self) -> None:
        """Compute and store the pressure profile."""
        self._p = self._p_norm * self._p0

    def _compute_density(self) -> None:
        """Compute and store the mass density profile."""
        self._rho = self._rho_norm * self._rho0

    def _compute_structure_factors(self) -> None:
        """Computes structure factors based on the dimensionless attributes of a polytrope structure file."""
        # compute and store the dtheta ratio, which is used to compute structure factors
        self._dt_ratio = self._dtheta / self._dtheta[-1]
        # compute and store the dtheta / theta ratio
        self._dt_t_ratio = self._dtheta / self._theta
        # V_2 factor
        self._compute_v_2()
        # V factor
        self._compute_v()
        # U factor
        self._compute_u()
        # c1 factor
        self._compute_c1()
        # As factor
        self._compute_as()
        # Mr factor (radial enclosed mass profile)
        self._compute_mr()
        # g factor
        self._compute_g()

    def _compute_v_2(self) -> None:
        """Computes and stores the V_2 structure factor."""
        self._v_2 = (
            -1.0 * (self._n + 1.0) * self._dt_t_ratio * (self._z_s / self._x)
        )

    def _compute_v(self) -> None:
        """Computes and stores the V structure factor."""
        self._v = -self._z * (self._n + 1.0) * self._dt_t_ratio

    def _compute_u(self) -> None:
        """Computes and stores the U structure factor."""
        self._u = -self._z * (self._theta**self._n) / self._dtheta

    def _compute_c1(self) -> None:
        """Computes and stores the c1 structure factor."""
        self._c1 = self._x / self._dt_ratio

    def _compute_as(self) -> None:
        """Computes and stores the As structure factor."""
        self._as = self._v * (
            (self._n / (self._n + 1.0)) - (1.0 / self._gamma1)
        )

    def _compute_mr(self) -> None:
        """Computes and stores the radial enclosed mass Mr."""
        self._mr = self._mass * self._dt_ratio * (self._x**2.0)

    def _compute_g(self) -> None:
        """Computes and stores the surface gravity g."""
        self._surf_grav = (
            self._G * self._mass / (self._radius**2.0)
        ) * self._dt_ratio

    def _compute_buoyancy_freq(self) -> None:
        """Computes the Brunt-Väisälä or buoyancy frequency."""
        self._brunt_squared = (self._surf_grav * self._as) / (
            self._x * self._radius
        )
