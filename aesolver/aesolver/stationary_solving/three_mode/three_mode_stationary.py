"""Python module containing class used to compute the stationary solutions of the AEs for the three-mode coupling :math: `A = B + C` (e.g. Van Beeck et al. 2023)

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# ensure type annotations are not evaluated
# but instead are handled as strings
from __future__ import annotations

# import statements
import logging
import numpy as np
from scipy.interpolate import splev, splrep

# import custom intra-package modules
from ..generic import StatSolve  # superclass

# type checking imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # import solver class for type checking
    from ...solver import QuadRotAESolver

    # import coupler class for type checking
    from ...coupling_coefficients import QCCR

    # import frequency handler class for type checking
    from ...frequency_handling import FH

    # import numpy typing
    import numpy.typing as npt


# set up logger
logger = logging.getLogger(__name__)


# create sub-class for the three-mode coupling A = B + C
class ThreeModeStationary(StatSolve):
    """Sub-class containing methods used to compute the stationary solutions for the three-mode coupling :math: `A = B + C`.

    Notes
    -----
    Support for values of the 'spin_inclination' and 'periastron_arg' other than None is experimental/work in progress.

    Parameters
    ----------
    precheck_satisfied: bool
        If True, all prechecks were satisfied and stationary solutions need to be computed. If False, stationary solutions do not need to be computed, as prechecks were not satisfied.
    freq_handler: FH
        Frequency handler object used for frequency conversions.
    hyperbolic : bool
        If True, the stationary solution is hyperbolic, if False, it is not.
    solver_object : QuadRotAESolver | None, optional
        Solver object from which data will be extracted in order to compute the stationary solutions.
    nr_modes : int, optional
        Number of modes to be included in the stationary computations; by default three.
    spin_inclination : float, optional
        Spin inclination angle of the observer in the rotating frame of the star, which is the angle between the spin/rotation axis and the line of sight (see e.g. Fuller & Lai, 2012). Input in radians. If None, the class ignores the value of the spin inclination angle (i.e. the corresponding :math: `\mu` value is set to 1.); by default None.
    periastron_arg: float, optional
        Argument of the periastron, only important for binary systems. If None, the class ignores the value of the azimuthal angle (i.e. it is set to 0); by default: None.
    """

    # attribute type declarations
    _pre: bool
    _freq_handler: 'FH'
    _coupler: 'QCCR | None'
    _cc_product: float | None
    _gamma: float
    _lin_drive_rates: 'npt.NDArray[np.float64]'
    _detuning: float
    _q: float
    _q_crit: float
    _mu_val_spin_inclination: None | float
    _az_ang_periastron: None | float
    _hyperbolic: bool
    _rel_stat_phase: float
    _a1: None | float
    _a2: None | float
    _a3: None | float
    _acrit: None | float
    _f1: None | float
    _f2: None | float
    _f3: None | float
    _rel_lum_phase: float
    _az_factors: 'npt.NDArray[np.float64]'
    _houghs_observer: 'npt.NDArray[np.float64]'

    def __init__(
        self,
        precheck_satisfied: bool,
        freq_handler: 'FH',
        hyperbolic: bool,
        solver_object: 'QuadRotAESolver | None' = None,
        nr_modes=3,
        spin_inclination: float | None = None,
        periastron_arg: float | None = None,
    ):
        # store whether the precheck is satisfied
        self._pre = precheck_satisfied
        # store a link to the frequency handler
        self._freq_handler = freq_handler
        # distinguish between cases where the precheck condition
        # is satisfied or not
        if self._pre and solver_object:
            # initialize superclass
            super().__init__(solver_object, nr_modes)
            # retrieve the QF products
            self._qf_products = self._freq_handler.quality_factor_products
            # compute and store the CC product
            self._get_cc_product()
        elif solver_object:
            self._coupler = solver_object._coupler
        else:
            self._coupler = None
        # retrieve the GYRE linear drive rates for the modes
        self._lin_drive_rates = self._freq_handler.driving_rates
        # compute gamma
        self._gamma = self._freq_handler.gamma_sum
        # compute the q factor and the critical q factor for this coupling
        self._compute_qs()
        # store the cosine of the spin inclination angle
        self._mu_val_spin_inclination = (
            None if (spin_inclination is None) else np.cos(spin_inclination)
        )
        # store the azimuthal angle of periastron
        self._az_ang_periastron = (
            None
            if (periastron_arg is None)
            else np.mod((0.5 * np.pi) - periastron_arg, 2.0 * np.pi)
        )
        # store whether the stationary point is hyperbolic
        self._hyperbolic = hyperbolic

    def __call__(self) -> dict:
        # distinguish between cases where the precheck condition
        # is satisfied or not
        if self._pre:
            # compute theoretical amplitudes and store them
            self._compute_theoretical_amps()
            # compute the complete disc-integral multiplication factors that do not take into account the direction of the observer (but take into account the mode Hough functions)
            super()._compute_disc_integral_multiplication_factors()
            # compute the possible observed luminosity fluctuations
            self._compute_luminosity_fluctuations()
            # compute the relative luminosity phase for this coupling
            self._compute_relative_luminosity_phase()
            # compute the relative stationary phase for this coupling
            self._compute_relative_stat_phase()
            # compute the Hough function of the observer at the spin inclination angle
            self._compute_observer_hough()
            # compute the azimuthal factors
            self._compute_azimuthal_factors()
            # compute the stationary luminosity phase
            self._compute_stationary_luminosity_phase()
            # create amplitude and luminosity fluc arrays
            amps = self._create_array_from_attrs('_a')
            flucs = self._create_array_from_attrs('_f') * self._houghs_observer
            # return the stationary solution information dictionary
            return {
                'equilibrium amplitudes': amps,
                'surface flux variations': flucs,
                'linear driving rates': self._lin_drive_rates,
                'gamma': self._gamma,
                'detunings': self._detuning,
                'q factor': self._q,
                'critical q factor': self._q_crit,
                'critical parent amplitude': self._acrit,
                'stationary relative phase': self._rel_stat_phase,
                'stationary relative luminosity phase': self._stat_rel_luminosity_phase,
                'hyperbolic': self._hyperbolic,
                'linearly stable': True,
            }
        else:
            # return fewer results in the solution information dictionary
            return {
                'equilibrium amplitudes': np.zeros((3,), dtype=np.float64),
                'surface flux variations': np.zeros((3,), dtype=np.float64),
                'linear driving rates': self._lin_drive_rates,
                'gamma': self._gamma,
                'detunings': self._detuning,
                'q factor': self._q,
                'critical q factor': self._q_crit,
                'critical parent amplitude': None,
                'stationary relative phase': None,
                'stationary relative luminosity phase': None,
                'hyperbolic': self._hyperbolic,
                'linearly stable': False,
            }

    def _get_cc_product(self) -> None:
        """Computes and stores the coupling coefficient product."""
        if TYPE_CHECKING:
            assert isinstance(self._coupler, 'QCCR')
        if (el_cc := self._coupler.eta_lee_coupling_coefficient) is None:
            self._cc_product = None
        else:
            self._cc_product = np.abs(el_cc) ** (2.0)

    def _compute_qs(self) -> None:
        """Computes the q factor and the critical q factor based on the linear frequencies from GYRE, and the corresponding linear driving rates."""
        # retrieve the theoretical detuning in the co-rotating frame from the coupler object
        _detuning_corot, _ = self._freq_handler.get_detunings()
        self._detuning = _detuning_corot[0]
        # compute the value of q (in the co-rotating frame)
        self._q = self._detuning / self._gamma
        # compute the value of the critical q (in the co-rotating frame)
        self._q_crit = self._detuning / self._lin_drive_rates[1:].sum()

    def _compute_theoretical_amps(self) -> None:
        """Computes the theoretical stationary equilibrium amplitudes for a parametric resonance that is stable (NOTE: according to the linear stability criterions: :math: `\gamma_{\boxplus} < 0` and the quartic criterion, not the full stability criterion).

        Also computes the theoretical critical parent amplitude for a parametric resonance.
        """
        # store None values if self._cc_product is None
        if self._cc_product is None:
            self._a1 = None
            self._a2 = None
            self._a3 = None
            self._acrit = None
        else:
            # compute the pre-factor, dependent on q, the QFs and the CC
            _pre_facs = 1.0 / (self._cc_product * self._qf_products)
            _pre_facs_q = (1.0 + self._q**2.0) * _pre_facs
            # compute the pre-factor for the critical parent amplitude
            _pre_facs_crit = (1.0 + self._q_crit**2.0) * _pre_facs[2]
            # compute the theoretical amplitudes
            self._a1 = np.sqrt(_pre_facs_q[2]) / 2.0
            self._a2 = np.sqrt(-_pre_facs_q[1]) / 2.0
            self._a3 = np.sqrt(-_pre_facs_q[0]) / 2.0
            # compute the critical parent amplitude
            self._acrit = np.sqrt(_pre_facs_crit) / 2.0

    def _compute_luminosity_fluctuations(self) -> None:
        """Computes the luminosity fluctuations. (Neglecting factors of line of sight of observer)"""
        # compute the luminosity fluctuations (Neglecting factors of line of sight of observer)
        for _i in self._mode_range:
            setattr(
                self,
                f'_f{_i}',
                getattr(self, f'_a{_i}')
                * np.absolute(self._disc_multi_factors[_i - 1]),
            )

    def _compute_relative_luminosity_phase(self) -> None:
        """Computes and stores the relative luminosity phases (DeltaL/L)_mode, internally stored as 'self._disc_multi_factors', and used in the comparison of relative phases."""
        # store intermediate array of complex angles
        # NOTE: the _disc_multi_factors are real-valued (see the superclass) for now, so these angles are zero.
        _angles = np.angle(self._disc_multi_factors)
        # compute and store the relative luminosity phase
        self._rel_lum_phase = _angles[0] - _angles[1:].sum()

    def _compute_relative_stat_phase(self) -> None:
        """Computes and stores the relative stationary phase based on the stored 'q' parameter."""
        self._rel_stat_phase = np.arctan(-1.0 / self._q)

    def _compute_stationary_luminosity_phase(self) -> None:
        """Compute the stationary luminosity phase difference that can be  compared to observations."""
        # compute the azimuthal factor sum
        _az_sum = self._az_factors[0] - self._az_factors[1:].sum()
        # compute the stationary relative luminosity phases
        self._stat_rel_luminosity_phase = (
            self._rel_stat_phase + self._rel_lum_phase + _az_sum
        )

    def _compute_observer_hough(self) -> None:
        """Compute the Hough function factor used to take into account the spin inclination angle of the observer in the rotating frame of the star.

        Notes
        -----
        Uses the Scipy B-spline representation of the Hough function to perform possibly needed extrapolation of the Hough function values near its end points.
        """
        # initialize the array storing the Hough function values for the observer
        self._houghs_observer = np.empty((self._nr_modes,), dtype=np.float64)
        # compute the Hough functions if the coupler object is present/loaded
        if self._coupler is not None:
            # store a link to the mu values
            _mu_vals = self._coupler._mu_values
            # loop through the relevant radial Hough functions for the modes, creating a B-spline representation for each, and interpolating for the Hough function value at the spin inclination angle
            for _nr in self._mode_range:
                # get the radial Hough function for the mode
                _mode_hough = getattr(self._coupler, f'_hr_mode_{_nr}')
                # construct the B-spline representation
                _b_spline_rep = splrep(_mu_vals, _mode_hough)
                # interpolate for the Hough function values, or SET TO 1 if we don't care about this factor
                self._houghs_observer[_nr - 1] = (
                    1.0
                    if self._mu_val_spin_inclination is None
                    else splev(self._mu_val_spin_inclination, _b_spline_rep)
                )

    def _compute_azimuthal_factors(self) -> None:
        """Compute the azimuthal factors for the modes associated with the position of the observer in the rotating frame of the star.

        Notes
        -----
        These factors are mostly, if not predominantly, important for binary systems, where the argument of periastron is used.
        """
        # check if the azimuthal angle is zero
        if (self._az_ang_periastron is None) or (
            self._az_ang_periastron == 0.0
        ):
            # store array of zeroes
            self._az_factors = np.zeros((self._nr_modes,), dtype=np.float64)
        else:
            # initialize empty array for the azimuthal factors
            self._az_factors = np.empty((self._nr_modes,), dtype=np.float64)
            # compute the azimuthal factors if the coupler object is present/loaded
            if self._coupler is not None:
                # loop over the relevant modes and obtain the azimuthal wavenumbers
                for _nr in self._mode_range:
                    # retrieve the azimuthal wavenumber
                    _m_val = getattr(self._coupler, f'_m_mode_{_nr}')
                    # store and compute the azimuthal wavenumber
                    self._az_factors[_nr - 1] = _m_val * self._az_ang_periastron
