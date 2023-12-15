"""Python module containing functions used to check the stability of the AE fixed point solutions for interaction between three distinct modes.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import module
import numpy as np
from multimethod import multimethod

# type checking imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # import numpy typing
    import numpy.typing as npt


# Generic condition for stability: gamma_1 + gamma_2 + gamma_3 < 0
@multimethod
def gamma_sum_stability(gammas: np.ndarray[np.float64]) -> bool:  # type: ignore
    """The generic condition for stability, solely based on the values of the linear driving/damping rates of the modes in the triad.

    Notes
    -----
    Should already be fulfilled during pre-check phase!

    Parameters
    ----------
    gammas : npt.NDArray[np.float64]
        Contains the linear driving/damping rates of the modes in the triad.

    Returns
    -------
    bool
        True if this stability condition is fulfilled. False if not.
    """
    return gammas.sum(axis=0) < 0.0


@multimethod
def gamma_sum_stability(gamma: float) -> bool:  # type: ignore
    """The generic condition for stability, solely based on the summed values of the linear driving/damping rates of the modes in the triad.

    Parameters
    ----------
    gamma : float
        Summed linear driving/damping rates of the modes in the triad.

    Returns
    -------
    bool
        True if this stability condition is fulfilled. False if not.
    """
    return gamma < 0.0


@multimethod
def gamma_sum_stability(gamma: list) -> bool:
    """The generic condition for stability, solely based on the summed values of the linear driving/damping rates of the modes in the triad.

    Parameters
    ----------
    gamma : list
        Contains the linear driving/damping rates of the modes in the triad.

    Returns
    -------
    bool
        True if this stability condition is fulfilled. False if not.
    """
    return sum(gamma) < 0.0


# Additional stability condition: see Van Beeck et al. (2022) and Dziembowski (1982).
def check_original_dziembowski_conditions(
    gammas: 'npt.NDArray[np.float64]', q: 'float | npt.NDArray[np.float64]'
) -> bool:
    """Performs the stability check according to the original formulation by Dziembowski (1982).

    Parameters
    ----------
    gammas : npt.NDArray[np.float64]
        Contains the linear driving/damping rates of the modes in the triad.
    q : float | npt.NDArray[np.float64]
        Ratio of the angular frequency detuning over the summed driving/damping rate.

    Returns
    -------
    bool
        True if this stability condition is fulfilled. False if not.
    """
    # compute specific gamma ratios
    rho2 = -gammas[1] / gammas[0]
    rho3 = -gammas[2] / gammas[0]
    rho23 = rho2 * rho3
    rhos = rho2 + rho3 - 1.0
    # compute the coefficients for the quartic q-polynomial
    coeffq0 = -((rhos) ** 3.0) - 2.0 * rho23
    coeffq2 = -12.0 * rho23 + rhos * (
        (rho2 - rho3) ** 2.0 + (rho2 + rho3) ** 2.0 + 2.0
    )
    coeffq4 = 3.0 * (
        -6.0 * rho23
        + rhos * ((rho2 - rho3) ** 2.0 + 2.0 * (rho2 + rho3) ** 2.0 + 1.0)
    )
    # compute the value of the quartic q-polynomial and assess and return whether the stability condition is fulfilled
    return (coeffq0 + (coeffq2 * (q**2.0)) + (coeffq4 * (q**4.0))) > 0.0


def check_corrected_dziembowski_conditions(
    gammas: 'npt.NDArray[np.float64]', q: 'float | npt.NDArray[np.float64]'
) -> bool:
    """Additional stability criterion that needs to be satisfied for the fixed point to be stable. Here we use the corrected Dziembowski criteria based on ratios of several quantities.

    Parameters
    ----------
    gammas : npt.NDArray[np.float64]
        Contains the linear driving/damping rates of the modes in the triad.
    q : float | npt.NDArray[np.float64]
        Ratio of the angular frequency detuning over the summed driving/damping rate.

    Returns
    -------
    bool
        True if this stability condition is fulfilled. False if not.
    """
    # compute the gamma ratios
    rho2 = -gammas[1] / gammas[0]
    rho3 = -gammas[2] / gammas[0]
    rho23 = rho2 * rho3
    rhos = 1.0 - rho2 - rho3
    # compute the coefficients for the quartic q-polynomial
    coeffq0 = (rhos) ** 3.0 - 2.0 * rho23
    coeffq2 = -12.0 * rho23 - rhos * (
        (rho2 - rho3) ** 2.0 + (rho2 + rho3) ** 2.0 + 2.0
    )
    coeffq4 = -18.0 * rho23 - (3.0 * rhos) * (
        (rhos**2.0) + 4.0 * (rho2 + rho3 - rho23)
    )
    # compute the value of the quartic q-polynomial and assess and return whether the stability condition is fulfilled
    return (coeffq0 + (coeffq2 * (q**2.0)) + (coeffq4 * (q**4.0))) > 0.0


def check_additional_condition_equivalent(
    gammas: 'npt.NDArray[np.float64]', q: 'float | npt.NDArray[np.float64]'
) -> bool:
    """Additional stability criterion that needs to be satisfied for the fixed point to be stable. This is equivalent to 'check_correct_dziembowski_conditions'.

    Parameters
    ----------
    gammas : npt.NDArray[np.float64]
        Contains the linear driving/damping rates of the modes in the triad.
    q : float | npt.NDArray[np.float64]
        Ratio of the angular frequency detuning over the summed driving/damping rate.

    Returns
    -------
    bool
        True if this stability condition is fulfilled. False if not.
    """
    # compute the summed and multiplied gamma values
    gamma_sum = gammas.sum()
    gamma_dot = gammas[0] * gammas[1] * gammas[2]
    gamma_box = gamma_dot / (gamma_sum**3.0)
    # compute intermediate values
    two_fac_sum = (
        gammas[0] * gammas[1] + gammas[0] * gammas[2] + gammas[1] * gammas[2]
    )
    gamma_star = two_fac_sum / (gamma_sum**2.0)
    # compute the coefficients for the quartic q-polynomial
    coeffq0 = (2.0 * gamma_box) - 1.0
    coeffq2 = 2.0 + (12.0 * gamma_box) - (4.0 * gamma_star)
    coeffq4 = (18.0 * gamma_box) + 3.0 - (12.0 * gamma_star)
    # compute the value of the quartic q-polynomial and assess and return whether
    # the stability condition is fulfilled
    return (coeffq0 + (coeffq2 * (q**2.0)) + (coeffq4 * (q**4.0))) > 0.0


def ratio_coefficients_q4_dziembowski(
    gammas: 'npt.NDArray[np.float64]'
) -> float:
    """Compute the ratio of the quartic stability polynomial's q^4 coefficients
    for the original Dziembowski (1982) formulation, and our corrected one.

    Parameters
    ----------
    gammas : npt.NDArray[np.float64]
        Contains the linear driving/damping rates of the modes in the triad.

    Returns
    -------
    float
        Corrected coefficient / original coefficient.
    """
    # compute the gamma ratios
    rho2 = -gammas[1] / gammas[0]
    rho3 = -gammas[2] / gammas[0]
    rho23 = rho2 * rho3
    rhos = rho2 + rho3 - 1.0  # Dziembowski
    rhos_corr = -rhos  # Van Beeck
    # compute the original Dziembowski (1982) coefficient
    coeff_original = 3.0 * (
        -6.0 * rho23
        + rhos * ((rho2 - rho3) ** 2.0 + 2.0 * (rho2 + rho3) ** 2.0 + 1.0)
    )
    # compute the corrected coefficient
    coeff_corrected = -18.0 * rho23 - (3.0 * rhos_corr) * (
        rhos_corr**2.0 + 4.0 * (rho2 + rho3 - rho23)
    )
    # return the ratio of both coefficients
    return coeff_corrected / coeff_original
