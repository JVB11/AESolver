"""Python module containing functions used to check the stability of the AE fixed point solutions for interaction between two modes (harmonic quadratic sum frequency resonance: :math: `\Omega_1 \approx 2 \Omega_2`).

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# typing import modules
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # only import numpy typing module when type checking
    import numpy.typing as npt


def compute_theta(gammas: 'npt.NDArray[np.float64]') -> float:
    """Computes the positive dimensionless theta variable for harmonic quadratic sum frequency resonances.

    Parameters
    ----------
    gammas : npt.NDArray[np.float64]
        Contains the linear driving/damping rates of the modes in the triad.

    Returns
    -------
    float
        The positive dimensionless theta variable.
    """
    return -gammas[1] / gammas[0]


def check_dziembowski_conditions_parametric(
    gammas: 'npt.NDArray[np.float64]', q: 'float | npt.NDArray[np.float64]'
) -> bool:
    """Implementation of the stability condition for the parametric harmonic sum frequency resonances, as described in Dziembowski(1982).

    Parameters
    ----------
    gammas : npt.NDArray[np.float64]
        Contains the linear driving/damping rates of the modes in the triad.
    q : float | npt.NDArray[np.float64]
        Value of the detuning-damping factor q.

    Returns
    -------
    bool
        True if the stability condition described in Dziembowski(1982) is fulfilled, False if not.
    """
    # compute the positive dimensionless theta variable
    theta = compute_theta(gammas)
    # compute the quotient used to enforce a condition on q^2
    denominator = 2.0 * theta * (theta - 1.0) + 1.0
    numerator = 2.0 * theta * (theta - 1.0) - 1.0
    q_squared_quotient = numerator / denominator
    # compute the minimum value of theta for stability
    min_theta: float = (1.0 + np.sqrt(3.0)) / 2.0
    # enforce the stability condition
    if TYPE_CHECKING:
        # ensure that the type checker is not fooled by the numpy number
        assert isinstance(q, float)
    return (theta > min_theta) and (q ** (2.0) > q_squared_quotient)


def check_dziembowski_conditions_direct(
    gammas: 'npt.NDArray[np.float64]'
) -> bool:
    """Implementation of the stability condition for the direct harmonic sum frequency resonances, as described in Dziembowski(1982).

    Parameters
    ----------
    gammas : npt.NDArray[np.float64]
        Contains the linear driving/damping rates of the modes in the triad.

    Returns
    -------
    bool
        True if the stability condition described in Dziembowski(1982) is fulfilled, False if not.
    """
    # enforce the stability condition
    return compute_theta(gammas) < 0.5
