"""Python module containing derived pre-check class used to perform specific pre-checks for quadratic coupling coefficients.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import logging
import numpy as np

# type checking imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any
    import numpy.typing as npt

# import intra-package modules to generate derived class
from ..generic import PreCheckQuadratic
from ...frequency_handling import FH

# import intra-package functions to perform stability checks
from ...stability_checker import (
    check_dziembowski_direct,
    check_dziembowski_parametric,
    check_van_beeck_conditions,
    gss,
)


# set up logger
logger = logging.getLogger(__name__)


# function library
def check_parametric(mode_info_dict: 'dict[str, Any]') -> bool:
    """Function that verifies that we are considering a
    parametric resonance scenario.

    Parameters
    ----------
    mode_info_dict : dict[str, Any]
        Contains the necessary driving rates to be checked.

    Returns
    -------
    bool
        True if we are considering a parametric resonance scenario.
        False otherwise.
    """
    # obtain the driving rates
    driving_rates = mode_info_dict['linear driving rates']
    # return check result
    return (
        (driving_rates[0] > 0.0)
        and (driving_rates[1] < 0.0)
        and (driving_rates[2] < 0.0)
    )


def check_direct(mode_info_dict: 'dict[str, Any]') -> bool:
    """Function for checking a direct resonance scenario.

    Parameters
    ----------
    mode_info_dict : dict[str, Any]
        Contains the necessary driving rates to be checked.

    Returns
    -------
    bool
        True if we are considering a direct resonance scenario.
        False otherwise.
    """
    # obtain the driving rates
    driving_rates = mode_info_dict['linear driving rates']
    # return check result
    return (
        (driving_rates[0] < 0.0)
        and (driving_rates[1] > 0.0)
        and (driving_rates[2] > 0.0)
    )


def check_driven(mode_info_dict: 'dict[str, Any]') -> bool:
    """Function for checking a driven resonance scenario.

    Parameters
    ----------
    mode_info_dict : dict[str, Any]
        Contains the necessary driving rates to be checked.

    Returns
    -------
    bool
        True if we are considering a driven direct resonance scenario.
        False otherwise.
    """
    # obtain the driving rates
    driving_rates = mode_info_dict['linear driving rates']
    # return check result
    return (driving_rates > 0.0).all()


# define the derived pre-check class
class PreCheckThreeModesQuadratic(PreCheckQuadratic):
    """Extension class containing methods to perform pre-checks for quadratic nonlinear coupling among gravito-inertial modes limited to isolated interactions (i.e. only three modes involved).

    Parameters
    ----------
    gyre_ad : list[dict[str, Any]]
        Contains the data files used for initiating the three-mode checks.
    freq_handler : FH
        The FrequencyHandler object used to read data for the pre-mode checks.
    triads : list[tuple[int, int, int]] | None, optional
        Denotes the mode numbers of the modes in the triplet/multiplet. If None, the stored mode numbers are [(1, 2, 3)]; by default None.
    conjugation : bool, optional
        Determines which of the modes require complex conjugation when computing the necessary integrals for AESolver. If None, perform no conjugation; by default None.
    two_mode_harmonic : bool, optional
        Determine whether prechecks should be performed for harmonic combinations or sum-frequency/difference combinations; by default False.
    """

    # attribute type declarations
    _lin_driv_rates: 'npt.NDArray[np.float64]'
    _q: 'float | npt.NDArray[np.float64]'
    _freq_handler: FH
    _two_mode_harmonic: bool

    def __init__(
        self,
        gyre_ad: 'list[dict[str, Any]]',
        freq_handler: FH,
        triads: list[tuple[int, int, int]] | None = None,
        conjugation: list[tuple[bool, bool, bool] | bool] | None = None,
        two_mode_harmonic: bool = False,
    ) -> None:
        # initialize the superclass object
        super().__init__(
            gyre_ad=gyre_ad,
            triads=triads,
            conjugation=conjugation,
            all_fulfilled=True,
        )
        # load additional mode information used for the specific checks
        # - linear driving rates
        self._lin_driv_rates = freq_handler.driving_rates.copy()
        # - frequency detuning over summed gamma
        if isinstance((my_q := freq_handler.q_factor), float):
            self._q = my_q
        else:
            # For some weird reason the type checker thinks an integer is a valid type???
            if TYPE_CHECKING:
                assert isinstance(my_q, 'np.ndarray')
            self._q = my_q.copy()
        # - store the frequency handler object
        self._freq_handler = freq_handler
        # - store whether we are dealing with a
        # two-mode harmonic resonance or three-mode resonance
        self._two_mode_harmonic = two_mode_harmonic

    # Check which resonance scenario is possible
    def _check_parametric(self) -> bool:
        """Verifies that we are considering a parametric resonance scenario.

        Returns
        -------
        bool
            True if we are considering a parametric resonance scenario. False otherwise.
        """
        return (
            (self._lin_driv_rates[0] > 0.0)
            and (self._lin_driv_rates[1] < 0.0)
            and (self._lin_driv_rates[2] < 0.0)
        )

    def _check_direct(self) -> bool:
        """Verifies that we are considering a direct resonance scenario.

        Returns
        -------
        bool
            True if we are considering a direct resonance scenario. False otherwise.
        """
        return (
            (self._lin_driv_rates[0] < 0.0)
            and (self._lin_driv_rates[1] > 0.0)
            and (self._lin_driv_rates[2] > 0.0)
        )

    def _check_driven(self) -> np.bool_:
        """Verifies that we are considering a driven resonance scenario.

        Returns
        -------
        np.bool_
            True if we are considering a driven resonance scenario. False otherwise.
        """
        return (self._lin_driv_rates > 0.0).all()

    # check whether a solution is linearly stable
    def _check_linearly_stable_fp(self) -> bool:
        """Checks whether the fixed point solution is linearly stable.

        Notes
        -----
        If the fixed point is not hyperbolic, the nonlinear stability of the fixed point cannot be checked through the linearization. However, the hyperbolicity condition cannot be checked until after mode coupling computations are performed. We therefore did not check hyperbolicity here! (The nonlinear stability of such points can be checked by other means, such as Lyapunov functions, see e.g. Guckenheimer & Holmes (1983).)

        Returns
        -------
        bool
            True if all (linear) stability conditions are fulfilled, False if not.
        """
        # First: check if we are dealing with parametric resonance conditions
        if self._check_parametric():
            # PARAMETRIC RESONANCE CONDITIONS
            # check if we are considering a two-mode harmonic sum resonance
            if self._two_mode_harmonic:
                # TWO MODE HARMONIC SUM RESONANCE
                return check_dziembowski_parametric(
                    self._lin_driv_rates, self._q
                )
            else:
                # THREE MODE SUM RESONANCE
                # - check the generic condition for stability: gamma_1 + gamma_2 + gamma_3 < 0
                generic_stab = gss(self._lin_driv_rates)
                # - check if the fixed point fulfills the additional stability conditions
                additional_stab = check_van_beeck_conditions(
                    self._lin_driv_rates, self._q
                )
                # return whether all conditions are fulfilled
                return generic_stab and additional_stab
        elif self._check_direct():
            # check if we are considering a two-mode harmonic sum resonance
            if self._two_mode_harmonic:
                return check_dziembowski_direct(self._lin_driv_rates)
            else:
                return False
        else:
            # NON-PARAMETRIC RESONANCE CONDITIONS ARE UNSTABLE FOR THREE-MODE SUM RESONANCES AND DRIVEN RESONANCE CONDITIONS ARE UNSTABLE FOR TWO-MODE HARMONIC SUM RESONANCES
            return False

    # perform the pre-check for this specific quadratic coupling case
    def check(self) -> bool:
        """Performs all necessary (quick) pre-checks before computing the coupling coefficient.

        Returns
        -------
        bool
            True if all pre-checks are satisfied. False otherwise.
        """
        # compute the generic check result (i.e. check if all selection rules are fulfilled)
        gen_check = super().generic_check()
        if TYPE_CHECKING:
            assert isinstance(gen_check, bool)
        # determine whether the fixed point is linearly stable
        # - nonlinear stability will have to be determined afterwards using the hyperbolicity
        # - condition!
        linear_stab_check = self._check_linearly_stable_fp()
        # return the specific check result
        return gen_check and linear_stab_check
