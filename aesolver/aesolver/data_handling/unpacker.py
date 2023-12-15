"""Python module in which data unpacking occurs and which is used as an intermediate medium to store these data in memory, before moving to more permanent storage on disk.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# ensure type annotations are not evaluated
# but instead are handled as strings
from __future__ import annotations

# import statements
import logging
import numpy as np

# import from intra-package modules
from ..stability_checker import gss
from ..pre_checks import check_direct, check_driven, check_parametric

# import modules used for typing
from typing import TYPE_CHECKING

# -- beating circular import errors
if TYPE_CHECKING:
    from .storer import NumericalSaver
    from ..mode_input_generator import InputGen
    from typing import Any


# set up logger
logger = logging.getLogger(__name__)


class UnPacker:
    """Python class containing methods used to unpack data and store them in memory, before saving to disk.

    Parameters
    ----------
    storage_object : NumericalSaver
        The object that makes use of the unpacked data to generate the file that will be saved to disk.
    """

    # attribute type declarations
    _store_poly: bool
    _stat_sol_data: 'list[dict[str, Any] | None]'
    _couple_data: list[tuple]
    _f_hand: list[tuple]
    _coupling_stats: list[tuple]
    _mode_info: 'InputGen'
    _nr_combos: int
    _nr_modes: int
    _num_int_method: str
    _use_cheby: bool
    _cheby_order_mul: int
    _cheby_blow_up: bool
    _cheby_blow_up_fac: float
    _pre_checks: np.ndarray
    _detunings: np.ndarray
    _gammas: np.ndarray
    _crit_parent_amp: np.ndarray
    _stat_sols_rel_phase: np.ndarray
    _stat_sols_rel_lum_phase: np.ndarray
    _q_factors: np.ndarray
    _critical_q_factors: np.ndarray
    _driving_rates: np.ndarray
    _eq_amps: np.ndarray
    _surf_flux_vars: np.ndarray
    _abs_ccs: np.ndarray
    _abs_nccs: np.ndarray
    _abs_etas: np.ndarray
    _adiabatic: np.ndarray
    _cc_terms: np.ndarray
    _norm_factors: np.ndarray
    _wu_rule: np.ndarray
    _inert_mode_freqs: np.ndarray
    _corot_mode_freqs: np.ndarray
    _inert_mode_omegas: np.ndarray
    _corot_mode_omegas: np.ndarray
    _inert_dimless_mode_freqs: np.ndarray
    _inert_dimless_mode_omegas: np.ndarray
    _corot_dimless_mode_freqs: np.ndarray
    _corot_dimless_mode_omegas: np.ndarray
    _inert_mode_freqs_nad: np.ndarray
    _corot_mode_freqs_nad: np.ndarray
    _inert_mode_omegas_nad: np.ndarray
    _corot_mode_omegas_nad: np.ndarray
    _inert_dimless_mode_freqs_nad: np.ndarray
    _inert_dimless_mode_omegas_nad: np.ndarray
    _corot_dimless_mode_freqs_nad: np.ndarray
    _corot_dimless_mode_omegas_nad: np.ndarray
    _spin_factors: np.ndarray
    _quality_factors: np.ndarray
    _surf_rot_f: np.ndarray
    _f_units: list[np.ndarray | float | complex]
    _mode_l: np.ndarray
    _mode_m: np.ndarray
    _mode_k: np.ndarray
    _mode_rad_orders: np.ndarray

    def __init__(self, storage_object: 'NumericalSaver') -> None:
        # store whether polytropic or MESA+GYRE data are unpacked
        self._store_poly = storage_object._store_poly
        # store the objects that store data to be saved
        # self._check_object = storage_object._check_object
        self._stat_sol_data = storage_object._stat_sol_data
        self._couple_data = storage_object._couple_data
        self._f_hand = storage_object._f_hand
        self._coupling_stats = storage_object._coupling_stats
        self._mode_info = storage_object._mode_info
        # store the nr. of combos
        self._nr_combos = storage_object._nr_combos
        self._nr_modes = storage_object._nr_modes
        # store the numerical integration method
        self._num_int_method = storage_object._num_int_method
        self._use_cheby = storage_object._use_cheby
        self._cheby_order_mul = storage_object._cheby_order_mul
        self._cheby_blow_up = storage_object._cheby_blow_up
        self._cheby_blow_up_fac = storage_object._cheby_blow_up_fac

    def __call__(self) -> None:
        # unpack the check data in a two-D numpy array
        self._unpack_check_data()
        # unpack the stationary solution data in multiple arrays
        self._unpack_stationary_solution_data()
        # unpack the mode coupling data in multiple arrays
        self._unpack_mode_coupling_data()
        # unpack mode statistics
        self._unpack_mode_stats()
        # unpack the mode frequency data in multiple arrays
        self._unpack_mode_frequency_data()
        # unpack the mode information
        self._unpack_mode_information()

    # unpack check data
    def _unpack_check_data(self) -> None:
        """Method that unpacks the check data into a two-dimensional boolean array.

        Notes
        -----
        Stores the following data array:

        pre_checks : np.ndarray[bool]
            Two-dimensional array containing the pre-check data.
        """
        # pre-checks do not matter for polytropic data: ONLY store when saving MESA+GYRE models
        if not self._store_poly:
            # - initialize numpy data array that contains information on the pre-checks:
            # col 1: check_parametric
            # col 2: check_direct
            # col 3: check_driven_direct
            # col 4: check_gamma
            # col 5: check linear stability
            # col 6: check_hyperbolicity
            self._pre_checks = np.empty((self._nr_combos, 6), dtype=np.bool_)
            # - fill the columns of the data array
            for _i, _stat_sol in enumerate(self._stat_sol_data):
                if TYPE_CHECKING:
                    assert _stat_sol
                self._pre_checks[_i, 0] = check_parametric(_stat_sol)
                self._pre_checks[_i, 1] = check_direct(_stat_sol)
                self._pre_checks[_i, 2] = check_driven(_stat_sol)
                self._pre_checks[_i, 3] = gss(_stat_sol['gamma'])
                self._pre_checks[_i, 4] = _stat_sol['linearly stable']
                self._pre_checks[_i, 5] = _stat_sol[
                    'hyperbolic'
                ]  # not really a pre-check

    # unpack stationary solution data
    def _unpack_stationary_solution_data(self) -> None:
        """Method that unpacks the stationary solution data into multiple arrays stored in this object/instance."""
        # STATIONARY SOLUTION DATA NOT IMPORTANT FOR POLYTROPES
        # - ONLY store this information for MESA+GYRE models
        if not self._store_poly:
            # - initialize numpy arrays that contain information on stationary solution data. This includes relative mode data, but does not include mode-specific data
            self._detunings = np.empty((self._nr_combos,), dtype=np.float64)
            self._gammas = np.empty_like(self._detunings)
            self._crit_parent_amp = np.empty_like(self._detunings)
            self._stat_sols_rel_phase = np.empty_like(self._detunings)
            self._stat_sols_rel_lum_phase = np.empty_like(self._detunings)
            self._q_factors = np.empty_like(self._detunings)
            self._critical_q_factors = np.empty_like(self._detunings)
            # - initialize numpy arrays that contain mode-specific data
            self._driving_rates = np.empty(
                (self._nr_combos, self._nr_modes), dtype=np.float64
            )
            self._eq_amps = np.empty_like(self._driving_rates)
            self._surf_flux_vars = np.empty_like(self._driving_rates)
            # - unpack and store the actual values
            for _i, _stat_sol in enumerate(self._stat_sol_data):
                if TYPE_CHECKING:
                    assert _stat_sol
                # store stationary solution data:
                # relative/mode triplet quantities
                self._detunings[_i] = _stat_sol['detunings']
                self._gammas[_i] = _stat_sol['gamma']
                self._crit_parent_amp[_i] = _stat_sol[
                    'critical parent amplitude'
                ]
                self._stat_sols_rel_phase[_i] = _stat_sol[
                    'stationary relative phase'
                ]
                self._stat_sols_rel_lum_phase[_i] = _stat_sol[
                    'stationary relative luminosity phase'
                ]
                self._q_factors[_i] = _stat_sol['q factor']
                self._critical_q_factors[_i] = _stat_sol['critical q factor']
                # store stationary solution data: mode-specific
                self._driving_rates[_i, :] = _stat_sol['linear driving rates']
                self._eq_amps[_i, :] = _stat_sol['equilibrium amplitudes']
                self._surf_flux_vars[_i, :] = _stat_sol[
                    'surface flux variations'
                ]

    # unpack mode coupling data
    def _unpack_mode_coupling_data(self) -> None:
        """Method that unpacks the mode coupling data into multiple arrays stored in this object/instance."""
        # - initialize numpy arrays that contain - information on mode coupling data
        # mode-triad-specific
        self._abs_ccs = np.empty((self._nr_combos,), dtype=np.float64)
        self._abs_nccs = np.empty_like(self._abs_ccs)
        self._abs_etas = np.empty_like(self._abs_ccs)
        self._adiabatic = np.empty((self._nr_combos,), dtype=np.bool_)
        self._cc_terms = np.empty((self._nr_combos, 4), dtype=np.float64)
        # mode-specific
        self._norm_factors = np.empty(
            (self._nr_combos, self._nr_modes), dtype=np.float64
        )
        # - unpack and store the actual values
        for _i, _mode_dat in enumerate(self._couple_data):
            # store coupling coefficients
            self._abs_ccs[_i] = _mode_dat[0]
            self._abs_nccs[_i] = _mode_dat[1]
            self._abs_etas[_i] = _mode_dat[2]
            # store coupling coefficient terms
            self._cc_terms[_i, :] = _mode_dat[8]
            # store normalization factors for the modes (radial eigenfunction norm)
            self._norm_factors[_i, :] = _mode_dat[5]
            # store boolean value that indicates whether quantities were computed adiabatically
            self._adiabatic[_i] = _mode_dat[6]

    # unpack frequency data
    def _unpack_mode_frequency_data(self) -> None:
        """Method that unpacks the mode frequency data into multiple arrays stored in this object/instance."""
        # - initialize numpy arrays that contain mode frequency information
        # ADIABATIC mode frequencies
        self._inert_mode_freqs = np.empty(
            (self._nr_combos, self._nr_modes), dtype=np.float64
        )
        self._corot_mode_freqs = np.empty_like(self._inert_mode_freqs)
        self._inert_mode_omegas = np.empty_like(self._inert_mode_freqs)
        self._corot_mode_omegas = np.empty_like(self._inert_mode_freqs)
        self._inert_dimless_mode_freqs = np.empty_like(self._inert_mode_freqs)
        self._corot_dimless_mode_freqs = np.empty_like(self._inert_mode_freqs)
        self._inert_dimless_mode_omegas = np.empty_like(self._inert_mode_freqs)
        self._corot_dimless_mode_omegas = np.empty_like(self._inert_mode_freqs)
        # NON-ADIABATIC mode frequencies: only for MESA+GYRE models!
        if not self._store_poly:
            self._inert_mode_freqs_nad = np.empty_like(self._inert_mode_freqs)
            self._corot_mode_freqs_nad = np.empty_like(self._inert_mode_freqs)
            self._inert_mode_omegas_nad = np.empty_like(self._inert_mode_freqs)
            self._corot_mode_omegas_nad = np.empty_like(self._inert_mode_freqs)
            self._inert_dimless_mode_freqs_nad = np.empty_like(
                self._inert_mode_freqs
            )
            self._corot_dimless_mode_freqs_nad = np.empty_like(
                self._inert_mode_freqs
            )
            self._inert_dimless_mode_omegas_nad = np.empty_like(
                self._inert_mode_freqs
            )
            self._corot_dimless_mode_omegas_nad = np.empty_like(
                self._inert_mode_freqs
            )
        # spin factors
        self._spin_factors = np.empty_like(self._inert_mode_freqs)
        # quality factors: only for MESA+GYRE models!
        if not self._store_poly:
            self._quality_factors = np.empty_like(self._inert_mode_freqs)
        # surface rotation frequency (based on corot frame freqs)
        self._surf_rot_f = np.empty((self._nr_combos,), dtype=np.float64)
        # store info on frequency units
        # -> create intermediate string list,
        # -> this will be handled in the submodules
        self._f_units = []
        # - perform the unpacking and store the data
        # --- polytropic information
        if self._store_poly:
            for _i, _f_inf in enumerate(self._f_hand):
                # store ADIABATIC frequencies
                self._inert_mode_freqs[_i, :] = _f_inf[0]
                self._corot_mode_freqs[_i, :] = _f_inf[1]
                self._inert_mode_omegas[_i, :] = _f_inf[2]
                self._corot_mode_omegas[_i, :] = _f_inf[3]
                self._inert_dimless_mode_freqs[_i, :] = _f_inf[4]
                self._corot_dimless_mode_freqs[_i, :] = _f_inf[5]
                self._inert_dimless_mode_omegas[_i, :] = _f_inf[6]
                self._corot_dimless_mode_omegas[_i, :] = _f_inf[7]
                # store spin factors
                self._spin_factors[_i, :] = _f_inf[8]
                # store surface rotation frequency
                self._surf_rot_f[_i] = _f_inf[9]
                # append the unit information
                self._f_units.append(_f_inf[10])
        # --- MESA+GYRE information
        else:
            for _i, _f_inf in enumerate(self._f_hand):
                # store ADIABATIC frequencies
                self._inert_mode_freqs[_i, :] = _f_inf[0]
                self._corot_mode_freqs[_i, :] = _f_inf[1]
                self._inert_mode_omegas[_i, :] = _f_inf[2]
                self._corot_mode_omegas[_i, :] = _f_inf[3]
                self._inert_dimless_mode_freqs[_i, :] = _f_inf[4]
                self._corot_dimless_mode_freqs[_i, :] = _f_inf[5]
                self._inert_dimless_mode_omegas[_i, :] = _f_inf[6]
                self._corot_dimless_mode_omegas[_i, :] = _f_inf[7]
                # store spin factors
                self._spin_factors[_i, :] = _f_inf[8]
                # store quality factors
                self._quality_factors[_i, :] = _f_inf[9]
                # store surface rotation frequency
                self._surf_rot_f[_i] = _f_inf[10]
                # append the unit information
                self._f_units.append(_f_inf[11])
                # store the NON-ADIABATIC frequencies
                self._inert_mode_freqs_nad[_i, :] = _f_inf[12]
                self._corot_mode_freqs_nad[_i, :] = _f_inf[13]
                self._inert_mode_omegas_nad[_i, :] = _f_inf[14]
                self._corot_mode_omegas_nad[_i, :] = _f_inf[15]
                self._inert_dimless_mode_freqs_nad[_i, :] = _f_inf[16]
                self._corot_dimless_mode_freqs_nad[_i, :] = _f_inf[17]
                self._inert_dimless_mode_omegas_nad[_i, :] = _f_inf[18]
                self._corot_dimless_mode_omegas_nad[_i, :] = _f_inf[19]

    def _unpack_mode_stats(self) -> None:
        """Method that unpacks the mode statistics."""
        # initialize numpy array for Wu et al. (2001) criterion
        self._wu_rule = np.empty((self._nr_combos,), dtype=np.int64)
        # - unpack the data and store it
        for _i, _couple_stat in enumerate(self._coupling_stats):
            self._wu_rule[_i] = _couple_stat[0]

    def _unpack_mode_information(self) -> None:
        """Method that unpacks the (GYRE) mode information: the quantum numbers."""
        # store the spherical degree, azimuthal order
        # and meridional order
        self._mode_l = np.array(self._mode_info.mode_l, dtype=np.int32)
        self._mode_m = np.array(self._mode_info.mode_m, dtype=np.int32)
        self._mode_k = self._mode_l - np.abs(self._mode_m)
        # store mode radial orders
        self._mode_rad_orders = np.array(
            self._mode_info.combinations_radial_orders, dtype=np.int32
        )
