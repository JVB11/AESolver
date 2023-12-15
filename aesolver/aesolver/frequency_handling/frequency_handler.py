"""Python module providing support class that computes frequencies and their derived detunings for coupling-coefficient-computing classes.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import logging
import sys
import numpy as np

# typing imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:   
    from typing import Any
    from collections.abc import Callable
    import numpy.typing as npt

# import intra-package enumeration module for frequency conversion
from .enumeration_files import EnumGYREFreqConv, is_angular_frequency


# set up logger
logger = logging.getLogger(__name__)


# class definition of frequency-computing class
class FrequencyHandler:
    """Handles computations and storage of frequencies and their derived detunings for the multiplet combinations considered.

    Parameters
    ----------
    gyre_dicts : tuple[dict]
        Contains the loaded GYRE information dictionaries of the modes considered in the computation of the coupling coefficients.
    my_freq_units : str, optional
        Denotes which frequency units are to be used for frequency conversion; by default 'hz' (CGS units), Options: ['cpd', 'hz', 'muhz']
    conj_list : list[bool], optional
        Denotes which of the modes are complex-conjugated in the coupling coefficient expression in Van Beeck et al. (2022), that is, which modes have subscripts that are barred in the coupling coefficient symbol, and which are not; by default None, which stores the default configuration for the quadratic coupling coefficient, [False, False, False].
    """

    # attribute type declarations
    _requested_units: str
    _n_modes: int
    _mode_dicts: "tuple[dict[str, Any], ...]"
    _angular: bool
    _cffs: "list[Callable]"
    _ocfr: "Callable"
    _fcfr: "Callable"
    _ocfc: "Callable"
    _fcfc: "Callable"
    _om_req: bool
    _rps_rot_freq: float
    _rot_freq: float
    _corot_mode_freqs: np.ndarray
    _inert_mode_freqs: np.ndarray
    _corot_mode_omegas: np.ndarray
    _inert_mode_omegas: np.ndarray
    _spin_factors: np.ndarray
    _dimless_corot_omegas: np.ndarray
    _dimless_inert_omegas: np.ndarray
    _dimless_corot_freqs: np.ndarray
    _dimless_inert_freqs: np.ndarray
    _driving_rates: np.ndarray
    _quality_factors: np.ndarray
    _quality_factors_nad: np.ndarray
    _quality_factor_products: np.ndarray
    _quality_factor_products_nad: np.ndarray
    _gamma_sum: float
    _corot_mode_freqs_nad: np.ndarray
    _inert_mode_freqs_nad: np.ndarray
    _corot_mode_omegas_nad: np.ndarray
    _inert_mode_omegas_nad: np.ndarray
    _dimless_corot_omegas_nad: np.ndarray
    _dimless_inert_omegas_nad: np.ndarray
    _dimless_corot_freqs_nad: np.ndarray
    _dimless_inert_freqs_nad: np.ndarray

    def __init__(
        self, *gyre_dicts: "dict[str, Any]", my_freq_units: str = 'HZ'
    ) -> None:
        # run the initialization method
        self.set_mode_info(*gyre_dicts, requested_units=my_freq_units)

    # set the mode information/initialize
    def set_mode_info(
        self, *gyre_mode_dicts: "dict[str, Any]", requested_units: str = 'HZ'
    ) -> None:
        """Sets the mode dictionaries, replacing them with passed values.

        Parameters
        ----------
        gyre_mode_dicts : tuple[dict, ...]
            Contains the loaded GYRE information dictionaries of the modes considered in the computation of the coupling coefficients.
        requested_units : str, optional
            Computes the output in this set of units. Default: 'HZ' (CGS units)
        """
        # store the requested units
        self._requested_units = requested_units
        # store the number of modes
        self._n_modes = len(gyre_mode_dicts)
        # unpack the gyre dictionaries
        self._mode_dicts = gyre_mode_dicts
        # store whether the input frequencies are angular or cyclic
        self._store_angular_unit_bool()
        # store the conversion factor functions
        self._store_cffs()
        # store the omega conversion functions
        self._retrieve_omega_freq_conversion_functions()
        # retrieve the rotation frequency
        self._retrieve_rotation_frequency()
        # retrieve the mode frequencies
        self._retrieve_mode_frequencies_az_orders()
        # compute the mode spin factors
        self._compute_mode_spin_factors()
        # compute the dimensionless frequencies
        self._compute_dimensionless_freqs(nonadiabatic=False)
        # compute the adiabatic frequency detunings
        self._compute_detunings(nad=False)

    # get the linear non-adiabatic information
    def set_mode_nonadiabatic_info(
        self, *gyre_nad_dicts: "dict[str, Any]"
    ) -> None:
        """Sets the non-adiabatic mode information, such as the linear driving/damping rates."""
        # check that the adiabatic and nonadiabatic GYRE dict lists are of the same length
        self._check_length_adiabatic_nonadiabatic(gyre_nad_dicts=gyre_nad_dicts)
        # get the nonadiabatic conversion functions
        _cffs_nad = self._get_nonadiabatic_conversion_functions(
            gyre_nad_dicts=gyre_nad_dicts
        )
        # retrieve and store the nonadiabatic frequencies and driving/damping rates
        self._get_nonadiabatic_frequencies_linear_driving_damping_rates(
            gyre_nad_dicts=gyre_nad_dicts, cffs_nad=_cffs_nad
        )
        # compute the additional non-adiabatic information
        # - quality factors
        self._compute_quality_factors()
        # - quality factor products
        self._compute_quality_factor_products()
        # - compute the gamma_sum factor
        self._compute_gamma_sum()
        # - compute the nonadiabatic dimensionless frequencies
        self._compute_dimensionless_freqs(nonadiabatic=True)
        # compute the non-adiabatic detuning
        self._compute_detunings(nad=True)
        # - compute the q and q1 factors
        self._compute_q_q1()

    @property
    def requested_units(self) -> str:
        """Returns the requested units string.

        Returns
        -------
        str
            Describes the units in which the frequencies are computed.
        """
        return self._requested_units

    @property
    def nr_of_modes(self) -> int:
        """Returns the number of modes for which frequencies are converted.

        Returns
        -------
        int
            Number of modes for which frequencies are converted to the requested type.
        """
        return self._n_modes

    @property
    def angular(self) -> bool:
        """Returns whether the requested frequency is angular.

        Returns
        -------
        bool
            True if the requested frequency is angular, False if the requested frequency is cyclic.
        """
        return self._angular

    @property
    def surf_rot_freq(self) -> float:
        """Returns the rotation frequency (dimensionalized).

        Returns
        -------
        float
            The dimensionalized rotation (angular or cyclic) frequency.
        """
        return self._rot_freq

    @property
    def surf_rot_angular_freq(self) -> float:
        """Returns the angular version of the rotation frequency (dimensionalized).

        Returns
        -------
        float
            The dimensionalized rotation angular frequency.
        """
        # verify that the requested frequency is angular
        try:
            my_angular = self._angular
        except AttributeError:
            my_angular = is_angular_frequency(
                requested_frequency_unit=self._requested_units
            )
        # return the angular version of the rotation frequency
        if my_angular:
            # no need for angular conversion
            return self._rot_freq
        else:
            # need for angularity conversion
            return self._rot_freq * 2.0 * np.pi

    @property
    def surf_rot_angular_freq_rps(self) -> float:
        """Returns the angular rotation frequency in radians per second.

        Returns
        -------
        float
            The rotation angular frequency in radians per second.
        """
        return self._rps_rot_freq

    @property
    def angular_conv_factor_from_rps(self) -> float:
        """Returns the multiplying conversion factor to obtain the requested frequency unit from an rps unit.

        Returns
        -------
        float
            Conversion factor for a RAD-PER-SECOND frequency into the angular version of the requested frequency unit.
        """
        # verify that the requested frequency is angular
        try:
            my_angular = self._angular
        except AttributeError:
            my_angular = is_angular_frequency(
                requested_frequency_unit=self._requested_units
            )
        # return the angular version of the conversion factor
        conv_factor = self._rot_freq / self._rps_rot_freq
        if my_angular:
            return conv_factor
        else:
            return conv_factor * 2.0 * np.pi

    @property
    def inert_mode_freqs(self) -> np.ndarray:
        """Returns the inertial frame mode frequencies.

        Returns
        -------
        np.ndarray
            The array containing frequencies in the inertial frame, from adiabatic calculations.
        """
        return self._inert_mode_freqs

    @property
    def inert_mode_omegas(self) -> np.ndarray:
        """Returns the inertial frame mode angular frequencies/omegas.

        Returns
        -------
        np.ndarray
            The array containing omegas in the inertial frame, from adiabatic calculations.
        """
        return self._inert_mode_omegas

    @property
    def corot_mode_freqs(self) -> np.ndarray:
        """Returns the corotating frame mode frequencies.

        Returns
        -------
        np.ndarray
            The array containing frequencies in the co-rotating frame, from adiabatic calculations.
        """
        return self._corot_mode_freqs

    @property
    def corot_mode_omegas(self) -> np.ndarray:
        """Returns the corotating frame mode angular frequencies/omegas.

        Returns
        -------
        np.ndarray
            The array containing omegas in the co-rotating frame, from adiabatic calculations.
        """
        return self._corot_mode_omegas

    @property
    def inert_mode_freqs_nad(self) -> np.ndarray:
        """Returns the non-adiabatic inertial frame mode frequencies.

        Returns
        -------
        np.ndarray
            The array containing frequencies in the inertial frame, from nonadiabatic calculations.
        """
        return self._inert_mode_freqs_nad

    @property
    def inert_mode_omegas_nad(self) -> np.ndarray:
        """Returns the non-adiabatic inertial frame mode angular frequencies/omegas.

        Returns
        -------
        np.ndarray
            The array containing omegas in the inertial frame, from nonadiabatic calculations.
        """
        return self._inert_mode_omegas_nad

    @property
    def corot_mode_freqs_nad(self) -> np.ndarray:
        """Returns the non-adiabatic corotating frame mode frequencies.

        Returns
        -------
        np.ndarray
            The array containing frequencies in the co-rotating frame, from nonadiabatic calculations.
        """
        return self._corot_mode_freqs_nad

    @property
    def corot_mode_omegas_nad(self) -> np.ndarray:
        """Returns the non-adiabatic corotating frame mode angular frequencies/omegas.

        Returns
        -------
        np.ndarray
            The array containing omegas in the co-rotating frame, from nonadiabatic calculations.
        """
        return self._corot_mode_omegas_nad

    @property
    def spin_factors(self) -> np.ndarray:
        """Returns the mode spin factors.

        Returns
        -------
        np.ndarray
            The array containing mode spin factors.
        """
        return self._spin_factors

    @property
    def dimless_corot_freqs(self) -> np.ndarray:
        """Returns the dimensionless frequencies in the corotating frame.

        Returns
        -------
        np.ndarray
            The array containing the values of the dimensionless frequencies in the co-rotating frame from nonadiabatic calculations.
        """
        return self._dimless_corot_freqs

    @property
    def dimless_inert_freqs(self) -> np.ndarray:
        """Returns the dimensionless frequencies in the inertial frame.

        Returns
        -------
        np.ndarray
            The array containing the values of the dimensionless frequencies in the inertial frame from adiabatic calculations.
        """
        return self._dimless_inert_freqs

    @property
    def dimless_corot_omegas(self) -> np.ndarray:
        """Returns the dimensionless angular frequencies in the corotating frame.

        Returns
        -------
        np.ndarray
            The array containing the values of the dimensionless omegas in the co-rotating frame from adiabatic calculations.
        """
        return self._dimless_corot_omegas

    @property
    def dimless_inert_omegas(self) -> np.ndarray:
        """Returns the dimensionless angular frequencies in the inertial frame.

        Returns
        -------
        np.ndarray
            The array containing the values of the dimensionless omegas in the inertial frame from adiabatic calculations.
        """
        return self._dimless_inert_omegas

    @property
    def dimless_corot_freqs_nad(self) -> np.ndarray:
        """Returns the dimensionless frequencies in the corotating frame.

        Returns
        -------
        np.ndarray
            The array containing the values of the dimensionless frequencies in the co-rotating frame from nonadiabatic calculations.
        """
        return self._dimless_corot_freqs_nad

    @property
    def dimless_inert_freqs_nad(self) -> np.ndarray:
        """Returns the dimensionless frequencies in the inertial frame.

        Returns
        -------
        np.ndarray
            The array containing the values of the dimensionless frequencies in the inertial frame from nonadiabatic calculations.
        """
        return self._dimless_inert_freqs_nad

    @property
    def dimless_corot_omegas_nad(self) -> np.ndarray:
        """Returns the dimensionless angular frequencies in the corotating frame.

        Returns
        -------
        np.ndarray
            The array containing the values of the dimensionless omegas in the co-rotating frame from nonadiabatic calculations.
        """
        return self._dimless_corot_omegas_nad

    @property
    def dimless_inert_omegas_nad(self) -> np.ndarray:
        """Returns the dimensionless angular frequencies in the inertial frame.

        Returns
        -------
        np.ndarray
            The array containing the values of the dimensionless omegas in the inertial frame from nonadiabatic calculations.
        """
        return self._dimless_inert_omegas_nad

    @property
    def driving_rates(self) -> np.ndarray:
        """Return the linear driving rates for the modes (angular!).

        Returns
        -------
        np.ndarray
            The array containing the values of the linear driving/damping rates of the modes.
        """
        return self._driving_rates

    @property
    def quality_factors(self) -> np.ndarray:
        """Return the quality factors of the modes.

        Returns
        -------
        np.ndarray
            The array containing the values of the quality factors of the modes.
        """
        return self._quality_factors

    @property
    def quality_factor_products(self) -> np.ndarray:
        """Return the products of the quality factors of the modes.

        Returns
        -------
        np.ndarray
            The array containing the products of quality factors.
        """
        return self._quality_factor_products

    @property
    def quality_factors_nad(self) -> np.ndarray:
        """Return the quality factors of the modes, computed using the nonadiabatic frequencies.

        Returns
        -------
        np.ndarray
            The array containing the values of the quality factors of the modes.
        """
        return self._quality_factors_nad

    @property
    def quality_factor_products_nad(self) -> np.ndarray:
        """Return the products of the quality factors of the modes, which were computed using the nonadiabatic frequencies.

        Returns
        -------
        np.ndarray
            The array containing the products of quality factors.
        """
        return self._quality_factor_products_nad

    @property
    def gamma_sum(self) -> float:
        """Return the gamma_sum (=sum linear driving rates) factor.

        Returns
        -------
        float
            The value of gamma_sum.
        """
        return self._gamma_sum

    @property
    def q_factor(self) -> "float | npt.NDArray[np.float64]":
        """Return the q factor (i.e. detuning-damping factor), which is the ratio: q = detuning / gamma_sum.

        Returns
        -------
        float | npt.NDArray[np.float64]
            The value of q (potentially in 0D numpy array format).
        """
        return self._q_fac

    @property
    def q1_factor(self) -> float:
        """Return the q1 factor (i.e. detuning-driving factor for parametric resonances), which is the ratio: q1 = detuning / gamma_1.

        Returns
        -------
        float
            The value of q1.
        """
        return self._q1_fac

    @property
    def q_factor_nad(self) -> float:
        """Return the q factor (i.e. detuning-damping factor), which is the ratio: q = detuning / gamma_sum for the nonadiabatic frequencies.

        Returns
        -------
        float
            The value of q using the nonadiabatic frequencies.
        """
        return self._q_fac_nad

    @property
    def q1_factor_nad(self) -> float:
        """Return the q1 factor (i.e. detuning-driving factor for parametric resonances), which is the ratio: q1 = detuning / gamma_1.

        Returns
        -------
        float
            The value of q1.
        """
        return self._q1_fac_nad

    @property
    def detuning(self) -> float:
        """Return the linear frequency detuning.

        Returns
        -------
        float
            The value of the linear frequency detuning.
        """
        return self._detuning

    @property
    def detuning_nad(self) -> float:
        """Return the linear nonadiabatic frequency detuning.

        Returns
        -------
        float
            The value of the linear frequency detuning.
        """
        return self._detuning_nad

    # get inertial and corotating frame frequency of requested number
    def get_inert_corot_f(self, my_f_nr: int = 1) -> np.ndarray:
        """Get the inertial and corotating frame frequency.

        Parameters
        ----------
        my_f_nr : int
            The number of the frequency for which you want to retrieve the frequencies.

        Returns
        -------
        np.ndarray
            Contains the frequencies in the co-rotating and inertial frame. First element = co-rotating frame, second = inertial.
        """
        return np.array(
            [
                self.corot_mode_freqs[my_f_nr - 1],
                self.inert_mode_freqs[my_f_nr - 1],
            ],
            dtype=np.float64,
        ).reshape((-1, 1))

    # get inertial and corotating frame frequency of requested number
    def get_inert_corot_f_nad(self, my_f_nr: int = 1) -> np.ndarray:
        """Get the nonadiabatic inertial and corotating frame frequency.

        Parameters
        ----------
        my_f_nr : int
            The number of the frequency for which you want to retrieve the frequencies.

        Returns
        -------
        np.ndarray
            Contains the frequencies in the co-rotating and inertial frame. First element = co-rotating frame, second = inertial.
        """
        return np.array(
            [
                self.corot_mode_freqs_nad[my_f_nr - 1],
                self.inert_mode_freqs_nad[my_f_nr - 1],
            ],
            dtype=np.float64,
        ).reshape((-1, 1))

    # get inertial and corotating frame omega of requested number
    def get_inert_corot_o(self, my_f_nr: int = 1) -> np.ndarray:
        """Get the inertial and corotating frame angular frequency.

        Parameters
        ----------
        my_f_nr : int
            The number of the frequency for which you want to retrieve the angular frequencies.

        Returns
        -------
        np.ndarray
            Contains the angular frequencies in the co-rotating and inertial frame. First element = co-rotating frame, second = inertial.
        """
        return np.array(
            [
                self.corot_mode_omegas[my_f_nr - 1],
                self.inert_mode_omegas[my_f_nr - 1],
            ],
            dtype=np.float64,
        ).reshape((-1, 1))

    # get inertial and corotating frame omega of requested number
    def get_inert_corot_o_nad(self, my_f_nr: int = 1) -> np.ndarray:
        """Get the nonadiabatic inertial and corotating frame angular frequency.

        Parameters
        ----------
        my_f_nr : int
            The number of the frequency for which you want to retrieve the angular frequencies.

        Returns
        -------
        np.ndarray
            Contains the angular frequencies in the co-rotating and inertial frame. First element = co-rotating frame, second = inertial.
        """
        return np.array(
            [
                self.corot_mode_omegas_nad[my_f_nr - 1],
                self.inert_mode_omegas_nad[my_f_nr - 1],
            ],
            dtype=np.float64,
        ).reshape((-1, 1))

    # get the detunings
    def get_detunings(self) -> np.ndarray:
        """Compute the (ANGULAR FREQUENCY) detunings for a nonlinear coupling.

        Notes
        -----
        At this moment only valid for SUM frequencies!!! TODO: generalize to allow input of formula

        Returns
        -------
        np.ndarray
            Contains the detunings in the co-rotating and inertial frame. First element = co-rotating frame, second = inertial.
        """
        # TODO: provide mappings for more generic detunings return the detunings (same for both frames) in a numpy array
        return np.array(
            [self._detuning, self._detuning], dtype=np.float64
        ).reshape((-1, 1))

    # get the detunings
    def get_detunings_nad(self) -> np.ndarray:
        """Compute the nonadiabatic (ANGULAR FREQUENCY) detunings for a nonlinear coupling.

        Notes
        -----
        At this moment only valid for SUM frequencies!!! TODO: generalize to allow input of formula

        Returns
        -------
        np.ndarray
            Contains the detunings in the co-rotating and inertial frame. First element = co-rotating frame, second = inertial.
        """
        # TODO: provide mappings for more generic detunings return the detunings (same for both frames) in a numpy array
        return np.array(
            [self._detuning_nad, self._detuning_nad], dtype=np.float64
        ).reshape((-1, 1))

    # get the dimensionless frequencies
    def get_dimless_freqs(self, my_f_nr: int = 1) -> np.ndarray:
        """Retrieve the dimensionless frequencies for a specific mode.

        Parameters
        ----------
        my_f_nr : int
            The number of the frequency for which you want to retrieve the dimensionless frequencies.

        Returns
        -------
        np.ndarray
            Contains the dimensionless frequencies in the co-rotating and inertial frame. First element = co-rotating frame, second = inertial.
        """
        return np.array(
            [
                self.dimless_corot_freqs[my_f_nr - 1],
                self.dimless_inert_freqs[my_f_nr - 1],
            ],
            dtype=np.float64,
        ).reshape((-1, 1))

    # get the dimensionless frequencies
    def get_dimless_freqs_nad(self, my_f_nr: int = 1) -> np.ndarray:
        """Retrieve the nonadiabatic dimensionless frequencies for a specific mode.

        Parameters
        ----------
        my_f_nr : int
            The number of the frequency for which you want to retrieve the dimensionless frequencies.

        Returns
        -------
        np.ndarray
            Contains the dimensionless frequencies in the co-rotating and inertial frame. First element = co-rotating frame, second = inertial.
        """
        return np.array(
            [
                self.dimless_corot_freqs_nad[my_f_nr - 1],
                self.dimless_inert_freqs_nad[my_f_nr - 1],
            ],
            dtype=np.float64,
        ).reshape((-1, 1))

    # get the dimensionless angular frequencies
    def get_dimless_omegas(self, my_f_nr: int = 1) -> np.ndarray:
        """Retrieve the dimensionless angular frequencies for a specific mode.

        Parameters
        ----------
        my_f_nr : int
            The number of the frequency for which you want to retrieve the dimensionless angular frequencies.

        Returns
        -------
        np.ndarray
            Contains the dimensionless omegas in the co-rotating and inertial frame. First element = co-rotating frame, second = inertial.
        """
        return np.array(
            [
                self.dimless_corot_omegas[my_f_nr - 1],
                self.dimless_inert_omegas[my_f_nr - 1],
            ],
            dtype=np.float64,
        ).reshape((-1, 1))

    # get the dimensionless angular frequencies
    def get_dimless_omegas_nad(self, my_f_nr: int = 1) -> np.ndarray:
        """Retrieve the dimensionless nonadiabatic angular frequencies for a specific mode.

        Parameters
        ----------
        my_f_nr : int
            The number of the frequency for which you want to retrieve the dimensionless angular frequencies.

        Returns
        -------
        np.ndarray
            Contains the dimensionless omegas in the co-rotating and inertial frame. First element = co-rotating frame, second = inertial.
        """
        return np.array(
            [
                self.dimless_corot_omegas_nad[my_f_nr - 1],
                self.dimless_inert_omegas_nad[my_f_nr - 1],
            ],
            dtype=np.float64,
        ).reshape((-1, 1))

    # store the frequency conversion functions
    def _store_cffs(self) -> None:
        """Stores the frequency conversion functions based on the requested frequency units (attribute)."""
        # store link to enumeration method
        enummeta = EnumGYREFreqConv.factor_from_dict
        # retrieve and store the frequency conversion functions
        self._cffs = [
            enummeta(_m_dict, self._requested_units, no_exit=True)
            for _m_dict in self._mode_dicts
        ]

    # store whether the requested frequency unit is angular (or cyclic)
    def _store_angular_unit_bool(self) -> None:
        """Stores whether the requested frequency unit is angular (or cyclic)."""
        # store a boolean that decides whether the requested units indicate angular frequency input (or cyclic frequency)
        self._angular = is_angular_frequency(
            requested_frequency_unit=self._requested_units
        )

    # retrieve the rotation frequency
    def _retrieve_rotation_frequency(self) -> None:
        """Retrieves and stores the rotation frequency."""
        # get the conversion function for the rotation frequency in rad / s
        _cff = EnumGYREFreqConv.factor_from_str(
            my_units=b'RAD_PER_SEC', my_requested_units=self._requested_units
        )
        if isinstance(_cff, KeyError):
            logger.error(
                'Requested units do not make sense for the rotation frequency. Now exiting.'
            )
            sys.exit()
        # get the dimensionalized rotation frequency in rad per seconds
        try:
            # get dimensionalized value from mode dictionaries
            self._rps_rot_freq = self._mode_dicts[0]['Omega_rot_dim'][-1]
        except KeyError:
            my_mode_dict = self._mode_dicts[0]
            # no dimensionalized Omega rot available:
            # - get necessary attributes for dimensionalization
            _G = my_mode_dict['G']
            _M = my_mode_dict['M_star']
            _R = my_mode_dict['R_star']
            # - perform dimensionalization
            self._rps_rot_freq = my_mode_dict['Omega_rot'][-1] * np.sqrt(
                (_G * _M) / (_R**3.0)
            )
        # convert to requested units and store
        self._rot_freq = _cff(self._rps_rot_freq)

    # retrieve omega/freq conversion functions
    def _retrieve_omega_freq_conversion_functions(self) -> None:
        """Retrieves and stores the omega/frequency conversion functions."""
        # based on the requested frequency unit (angular or cyclic), retrieve and store the conversion functions
        if self._angular:
            self._ocfr = lambda val: val.real[0]
            self._fcfr = lambda val: val.real[0] / (np.pi * 2.0)
            self._ocfc = lambda val: val[0]
            self._fcfc = lambda val: val[0] / (np.pi * 2.0)
            # store attribute that states that omegas are requested
            self._om_req = True
        else:
            self._ocfr = lambda val: val.real[0] * 2.0 * np.pi
            self._fcfr = lambda val: val.real[0]
            self._ocfc = lambda val: val[0] * 2.0 * np.pi
            self._fcfc = lambda val: val[0]
            # store attribute that states that omegas are NOT requested
            self._om_req = False

    # retrieve the mode frequencies from the GYRE dictionaries
    def _retrieve_mode_frequencies_az_orders(self) -> None:
        """Retrieves and store the mode frequencies from the GYRE dictionaries, as well as the azimuthal orders."""
        # initialize the numpy arrays used to store the mode frequencies
        self._corot_mode_freqs = np.empty((self._n_modes,), dtype=np.float64)
        self._inert_mode_freqs = np.empty_like(self._corot_mode_freqs)
        self._corot_mode_omegas = np.empty_like(self._corot_mode_freqs)
        self._inert_mode_omegas = np.empty_like(self._corot_mode_freqs)
        # retrieve and store the mode frequencies from the GYRE dictionaries
        # - loop over the GYRE dictionaries
        for _i, (_md, _cff) in enumerate(zip(self._mode_dicts, self._cffs)):
            # obtain the converted frequency
            _conv_freq = _cff(_md['freq'])
            # get the 'm' value (=> no complex conjugation needed)
            _m = _md['m']
            # convert to requested units
            _convert_freq_omega = self._ocfr(_conv_freq)
            _convert_freq_freq = self._fcfr(_conv_freq)
            # determine action based on stored frequency frame
            if _md['freq_frame'] == b'INERTIAL':
                # INERTIAL FRAME INPUT DATA
                # - store inertial mode freq and angular freq
                self._inert_mode_freqs[_i] = _convert_freq_freq.real
                self._inert_mode_omegas[_i] = _convert_freq_omega.real
                # - compute corotating mode freq
                # check if angular frequencies were requested or not
                if self._angular:
                    # ANGULAR FREQUENCY REQUESTED: NO NEED TO CONVERT ROT. FREQ TO ANGULAR FREQ. !
                    self._corot_mode_omegas[_i] = _convert_freq_omega.real - (
                        _m * self._rot_freq
                    )
                    self._corot_mode_freqs[_i] = self._corot_mode_freqs[_i] / (
                        2.0 * np.pi
                    )
                else:
                    # CYCLIC FREQUENCY REQUESTED: USE CYCLIC ROTATION FREQUENCY !
                    self._corot_mode_freqs[_i] = _convert_freq_freq.real - (
                        _m * self._rot_freq
                    )  # rot_freq is in requested cyclic units!
                    self._corot_mode_omegas[_i] = (
                        2.0 * np.pi
                    ) * self._corot_mode_freqs[_i]
            else:
                # COROTATING FRAME INPUT DATA
                # - store corotating mode freq
                self._corot_mode_freqs[_i] = _convert_freq_freq.real
                self._corot_mode_omegas[_i] = _convert_freq_omega.real
                # - compute inertial mode freq
                # check if angular frequencies were requested or not
                if self._angular:
                    # ANGULAR FREQUENCY REQUESTED: NO NEED TO CONVERT ROT. FREQ TO ANGULAR FREQ. !
                    self._inert_mode_omegas[_i] = _convert_freq_omega.real + (
                        _m * self._rot_freq
                    )
                    self._inert_mode_freqs[_i] = self._inert_mode_omegas[_i] / (
                        2.0 * np.pi
                    )
                else:
                    # CYCLIC FREQUENCY REQUESTED: USE CYCLIC ROTATION FREQUENCY !
                    self._inert_mode_freqs[_i] = _convert_freq_freq.real + (
                        _m * self._rot_freq
                    )
                    self._inert_mode_omegas[_i] = (
                        2.0 * np.pi
                    ) * self._inert_mode_freqs[_i]

    # retrieve the mode spin factors
    def _compute_mode_spin_factors(self) -> None:
        """Computes the mode spin factors based on the loaded mode frequencies, and the rotation frequency."""
        # store the spin factors based on computed frequencies
        # -- check first if angular frequencies were requested or not
        if self._angular:
            # no conversion of (angular) rotation frequency necessary to compute the spin factor
            self._spin_factors = 2.0 * self._rot_freq / self._corot_mode_omegas
        else:
            # conversion of (cyclic) rotation frequency necessary (to angular frequency) to compute the spin factor
            self._spin_factors = (
                4.0 * np.pi * self._rot_freq / self._corot_mode_omegas
            )

    # compute the dimensionless angular frequencies for the different modes
    def _compute_dimensionless_freqs(self, nonadiabatic: bool = False) -> None:
        """Computes the dimensionless angular and cyclic frequencies based on the loaded mode data.

        Parameters
        ----------
        nonadiabatic : bool, optional
            If True, compute the dimensionless frequencies based on the loaded non-adiabatic frequencies. If False, use the adiabatic frequencies; by default False.
        """
        # retrieve the unit conversion method for the dimensioning factor (which is always computed in 1/s or Hz (because of the cgs units!!!))
        _cfc = EnumGYREFreqConv.factor_for_dimensioning_factor(
            self._requested_units
        )
        # store local link to first mode dictionary containing stellar evolution model information
        _md = self._mode_dicts[0]
        # compute the correctly dimensioned dimensioning factor for frequencies in the requested frequency unit
        dim_factor = _cfc(
            np.sqrt((_md['R_star']) ** (3.0) / (_md['G'] * _md['M_star']))
        )  # IN REQUESTED UNITS! (not necessarily CYCLIC!!!)
        # non-adiabatic vs. adiabatic
        if nonadiabatic:
            # compute and store the dimensionless angular frequencies
            self._dimless_corot_omegas_nad = (
                self.corot_mode_omegas_nad * dim_factor
            )
            self._dimless_inert_omegas_nad = (
                self.inert_mode_omegas_nad * dim_factor
            )
            # compute and store the dimensionless cyclic frequencies
            self._dimless_corot_freqs_nad = (
                self.corot_mode_freqs_nad * dim_factor
            )
            self._dimless_inert_freqs_nad = (
                self.inert_mode_freqs_nad * dim_factor
            )
        else:
            # compute and store the dimensionless angular frequencies
            self._dimless_corot_omegas = self.corot_mode_omegas * dim_factor
            self._dimless_inert_omegas = self.inert_mode_omegas * dim_factor
            # compute and store the dimensionless cyclic frequencies
            self._dimless_corot_freqs = self.corot_mode_freqs * dim_factor
            self._dimless_inert_freqs = self.inert_mode_freqs * dim_factor

    # compute the detunings
    def _compute_detunings(self, nad: bool = False) -> None:
        """Computes the detunings in the co-rotating frame.

        Parameters
        ----------
        nad : bool, optional
            If True, compute the non-adiabatic detuning. If False, compute the adiabatic detuning; by default False.
        """
        if nad:
            # compute the non-adiabatic detuning
            self._detuning_nad = self._corot_mode_omegas_nad[
                0
            ] - self._corot_mode_omegas_nad[1:].sum(axis=0)
        else:
            # compute the adiabatic detuning
            self._detuning = self._corot_mode_omegas[
                0
            ] - self._corot_mode_omegas[1:].sum(axis=0)

    # check adiabatic and non-adiabatic units
    def _check_adiabatic_nonadiabatic_units(
        self, gyre_nad_dicts: "tuple[dict[str, Any], ...]"
    ) -> bool:
        """Checks whether the (frequency) units of the adiabatic and nonadiabatic GYRE data are the same.

        Parameters
        ----------
        gyre_nad_dicts : tuple[dict[str, Any], ...]
            Contains the nonadiabatic information from the GYRE calculations.

        Returns
        -------
        bool
            True if the (frequency) units of the adiabatic and nonadiabatic GYRE data are the same, False otherwise.
        """
        return all(
            [
                a['freq_units'] == n['freq_units']
                for a, n in zip(self._mode_dicts, gyre_nad_dicts)
            ]
        )

    # get nonadiabatic conversion functions
    def _get_nonadiabatic_conversion_functions(
        self, gyre_nad_dicts: "tuple[dict[str, Any], ...]"
    ) -> "list[Callable]":
        """Get the nonadiabatic conversion functions.

        Parameters
        ----------
        gyre_nad_dicts : tuple[dict[str, Any], ...]
            Contains the nonadiabatic information from the GYRE calculations.

        Returns
        -------
        list[Callable]
            Contains the nonadiabatic conversion functions.
        """
        # check if the units for adiabatic and nonadiabatic calculations are the same
        same_freq_units = self._check_adiabatic_nonadiabatic_units(
            gyre_nad_dicts=gyre_nad_dicts
        )
        # return the conversion functions
        if same_freq_units:
            # use same functions
            return self._cffs
        else:
            # obtain the additional conversion functions for nonadiabatic input
            enummeta_nad = EnumGYREFreqConv.factor_from_dict
            return [
                enummeta_nad(_m_dict_n, self._requested_units)
                for _m_dict_n in gyre_nad_dicts
            ]

    def _check_length_adiabatic_nonadiabatic(
        self, gyre_nad_dicts: "tuple[dict[str, Any], ...]"
    ) -> None:
        """Verify that lengths of nonadiabatic and adiabatic calculations are the same, or raise an error.

        Parameters
        ----------
        gyre_nad_dicts : tuple[dict[str, Any], ...]
            Contains the nonadiabatic information from the GYRE calculations.
        """
        # same length check
        same_len = len(gyre_nad_dicts) == len(self._mode_dicts)
        # raise error if necessary
        if not same_len:
            logger.error(
                'GYRE nonadiabatic dictionary list is not of the same length as the adiabatic one! Now exiting.'
            )
            sys.exit()

    def _preallocate_nonadiabatic_variables(self) -> None:
        """Preallocates arrays for storage of nonadiabatic variables."""
        # pre-allocate the array that will contain the driving rates
        self._driving_rates = np.empty((self._n_modes,), dtype=np.float64)
        # pre-allocate the arrays that will contain the non-adiabatic frequencies
        self._corot_mode_freqs_nad = np.empty_like(self._driving_rates)
        self._inert_mode_freqs_nad = np.empty_like(self._driving_rates)
        self._corot_mode_omegas_nad = np.empty_like(self._driving_rates)
        self._inert_mode_omegas_nad = np.empty_like(self._driving_rates)

    def _get_nonadiabatic_frequencies_linear_driving_damping_rates(
        self,
        gyre_nad_dicts: "tuple[dict[str, Any], ...]",
        cffs_nad: "list[Callable]",
    ) -> None:
        """Retrieve and store the nonadiabatic frequencies and linear driving/damping rates."""
        # preallocate nonadiabatic variables
        self._preallocate_nonadiabatic_variables()
        # obtain the frequencies and driving rates
        for _i, (_nad, _cff_n) in enumerate(zip(gyre_nad_dicts, cffs_nad)):
            # obtain the converted frequency
            _conv_freq = _cff_n(_nad['freq'])
            # get the converted 'm' value (=> taking into account complex conjugation)
            _m = _nad['m']
            # _m = PreCheckQuadratic._conjer_m(_conj, _nad['m'])
            # convert to requested units
            _convert_freq_omega = self._ocfc(_conv_freq)
            _convert_freq_freq = self._fcfc(_conv_freq)
            # determine action based on stored frequency frame
            if _nad['freq_frame'] == b'INERTIAL':
                # INERTIAL FRAME INPUT DATA
                # - store inertial mode freq and angular freq
                self._inert_mode_freqs_nad[_i] = _convert_freq_freq.real
                self._inert_mode_omegas_nad[_i] = _convert_freq_omega.real
                # - compute corotating mode freqs
                # check if angular frequencies were requested or not
                if self._angular:
                    # ANGULAR FREQUENCY REQUESTED: NO NEED TO CONVERT ROT. FREQ TO ANGULAR FREQ. !
                    self._corot_mode_omegas_nad[_i] = (
                        _convert_freq_omega.real - (_m * self._rot_freq)
                    )
                    self._corot_mode_freqs_nad[_i] = (
                        self._corot_mode_omegas_nad[_i] / (2.0 * np.pi)
                    )
                else:
                    # CYCLIC FREQUENCY REQUESTED: USE CYCLIC ROTATION FREQUENCY !
                    self._corot_mode_freqs_nad[_i] = _convert_freq_freq.real - (
                        _m * self._rot_freq
                    )
                    self._corot_mode_omegas_nad[_i] = (
                        2.0 * np.pi
                    ) * self._corot_mode_freqs_nad[_i]
                # - convert the inertial complex angular frequency to a rotating frame angular frequency (only matters for complex-valued surface rotation frequencies, but nonetheless this code check is more inclusive)
                # check if angular frequencies were requested
                if self._angular:
                    # ANGULAR FREQUENCY REQUESTED: NO NEED TO CONVERT ROT. FREQ TO ANGULAR FREQ. !
                    _rot_omega_complex = _convert_freq_omega - (
                        _m * self._rot_freq
                    )
                else:
                    # CYCLIC FREQUENCY REQUESTED: CONVERT TO ANGULAR ROTATION FREQUENCY !
                    _rot_omega_complex = _convert_freq_omega - (
                        _m * self._rot_freq * (2.0 * np.pi)
                    )
                # - store the driving rate (rotating frame)
                self._driving_rates[_i] = _rot_omega_complex.imag
            else:
                # ROTATING FRAME INPUT DATA
                # - store corotating mode freq and angular freq
                self._corot_mode_freqs_nad[_i] = _convert_freq_freq.real
                self._corot_mode_omegas_nad[_i] = _convert_freq_omega.real
                # - compute inertial mode freqs
                # check if angular frequencies were requested or not
                if self._angular:
                    # ANGULAR FREQUENCY REQUESTED: NO NEED TO CONVERT ROT. FREQ TO ANGULAR FREQ. !
                    self._inert_mode_omegas_nad[_i] = (
                        _convert_freq_omega.real + (_m * self._rot_freq)
                    )
                    self._inert_mode_freqs_nad[_i] = (
                        self._inert_mode_omegas_nad[_i] / (2.0 * np.pi)
                    )
                else:
                    # CYCLIC FREQUENCY REQUESTED: USE CYCLIC ROTATION FREQUENCY !
                    self._inert_mode_freqs_nad[_i] = _convert_freq_freq.real + (
                        _m * self._rot_freq
                    )
                    self._inert_mode_omegas_nad[_i] = (
                        2.0 * np.pi
                    ) * self._inert_mode_freqs_nad[_i]
                # - store the driving rate (rotating frame; ANGULAR!)
                self._driving_rates[_i] = _convert_freq_omega.imag

    # compute the quality factors of the modes
    def _compute_quality_factors(self) -> None:
        """Computes the quality factors for the different modes."""
        # store the quality factors
        # -- using adiabatic frequencies
        self._quality_factors = self._corot_mode_omegas / self._driving_rates
        # -- using non-adiabatic frequencies
        self._quality_factors_nad = (
            self._corot_mode_omegas_nad / self._driving_rates
        )

    # compute the products of the quality factors of the modes
    def _compute_quality_factor_products(self) -> None:
        """Computes the products of the quality factors."""
        # store the outer product of the quality factors in a local variable
        _q_oprod = self._quality_factors * self._quality_factors[:, np.newaxis]
        _q_oprod_nad = (
            self._quality_factors_nad * self._quality_factors_nad[:, np.newaxis]
        )
        # store the products of the quality factors
        self._quality_factor_products = _q_oprod[
            np.tril_indices(_q_oprod.shape[0], k=-1)
        ]
        self._quality_factor_products_nad = _q_oprod_nad[
            np.tril_indices(_q_oprod_nad.shape[0], k=-1)
        ]

    # compute the gamma_sum (= sum of all linear driving rates) factor
    def _compute_gamma_sum(self) -> None:
        """Computes the gamma_sum (=sum of all linear driving rates) factor."""
        self._gamma_sum = self._driving_rates.sum(axis=0)

    # compute the q and q1 factors
    def _compute_q_q1(self) -> None:
        """Computes the q and q_1 factors for the given mode triad."""
        # non-adiabatic
        self._q_fac_nad = self._detuning_nad / self._gamma_sum
        self._q1_fac_nad = self._detuning_nad / self._driving_rates[0]
        # adiabatic
        self._q_fac = self._detuning / self._gamma_sum
        self._q1_fac = self._detuning / self._driving_rates[0]
