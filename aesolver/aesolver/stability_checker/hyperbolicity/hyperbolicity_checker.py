"""Checks whether the fixed point (FP) is hyperbolic. If this is the case, then, according to the Hartman-Grobman theorem, the local geometry of the nonlinear problem is homeomorphic to that of the linearized problem. Hence, if the FP is hyperbolic, the linearization can be used to probe the stability of the FP.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import numpy as np
import logging

# type checking imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # numpy typing
    import numpy.typing as npt


# set up logger
logger = logging.getLogger(__name__)


def check_hyperbolicity_of_fp(my_jac: 'npt.NDArray[np.float64]') -> bool:
    """Checks the hyperbolicity of the fixed point, by computing the eigenvalues of the jacobian of the system of AEs in that fixed point.

    Notes
    -----
    A fixed point is hyperbolic if the real parts of the eigenvalues of the jacobian of the system of AEs evaluated in the fixed point are not equal to zero.

    Parameters
    ----------
    my_jac : npt.NDArray[np.float64]
        Jacobian of the system of AEs evaluated in the fixed point.

    Returns
    -------
    bool
        True if the fixed point is hyperbolic, False if not.
    """
    try:
        # compute the eigenvalues
        eigs = np.linalg.eigvals(my_jac)
        # compute the determinant of the jacobian
        det = np.linalg.det(my_jac)
        # check if their real parts are not equal to zero and return the hyperbolicity check result
        return (np.real(eigs) != 0.0).all() and (det != 0.0)
    except np.linalg.LinAlgError:
        logger.exception(
            f'The jacobian matrix {my_jac} contains infinite or NaN values. This probably relates to input frequency/driving rate information!'
        )
        # return a False value when these values appear!
        return False


def jacobian_in_fp_real_four_ae_system(
    gammas: 'npt.NDArray[np.float64]',
    omegas: 'npt.NDArray[np.float64]',
    q: 'float | npt.NDArray[np.float64]',
    eta: 'float | npt.NDArray[np.float64] | list[float]',
) -> 'npt.NDArray[np.float64]':
    """Compute the jacobian matrix in the fixed point for a system of four real-valued AEs.

    Parameters
    ----------
    gammas : npt.NDArray[np.float64]
        Linear driving/damping rates of the mode triad.
    omegas : npt.NDArray[np.float64]
        Linear angular frequencies of the mode triad.
    q : float | npt.NDArray[np.float64]
        Ratio of the angular frequency detuning over the summed driving/damping rate.
    eta : float | npt.NDArray[np.float64]
        Coupling coefficient.

    Returns
    -------
    jac : npt.NDArray[np.float64]
        The jacobian matrix in the fixed point.
    """
    if isinstance(eta, list):
        eta = np.array(eta)
    # initialize the jacobian array
    jac = np.zeros((4, 4), dtype=np.float64)
    # store intermediate values
    inv_quality = gammas / omegas  # gx / ox
    sqrt12 = np.sqrt(-inv_quality[0] * inv_quality[1])  # sqrt((-g1*g2)/(o1*o2))
    sqrt13 = np.sqrt(-inv_quality[0] * inv_quality[2])  # sqrt((-g1*g3)/(o1*o3))
    sqrt23 = np.sqrt(inv_quality[1] * inv_quality[2])  # sqrt((g2*g3)/(o2*o3))
    qsqrt = q * np.sqrt(1.0 + (q**2.0))  # q*sqrt(1+q^2)
    deta = 2.0 * eta  # 2*eta
    gplus = gammas.sum()  # gplus
    qf = qsqrt / deta  # q*sqrt(1+q^2) / (2*eta)
    qsq_over_qf = q**2.0 / qf  # (2*eta*q) / sqrt(1+q^2)
    # fill up the jacobian elements
    jac[np.diag_indices(3)] = gammas  # (g1,g2,g3)
    jac[0, 1] = (
        omegas[0] * sqrt12
    )  # o1 * sqrt((-g1*g2)/(o1*o2)) = sqrt(-g1*o1*g2/o2)
    jac[0, 2] = (
        omegas[0] * sqrt13
    )  # o1 * sqrt((-g1*g3)/(o1*o3)) = sqrt(-g1*o1*g3/o3)
    jac[0, 3] = (
        qf * gammas[0] * sqrt23
    )  # (q*sqrt(1+q^2)/(2*eta))*(g1)*sqrt((g2*g3)/(o2*o3)) = (q*g1/(2*eta))*sqrt((1+q^2)*g2*g3/(o2*o3))
    jac[1, 0] = (
        -omegas[1] * sqrt12
    )  # -o2 * sqrt((-g1*g2)/(o1*o2)) = -sqrt(-g1*o2*g2/o1)
    jac[1, 2] = (
        -omegas[1] * sqrt23
    )  # -o2 * sqrt((g2*g3)/(o2*o3)) = -sqrt(g2*o3*g3/o2)
    jac[1, 3] = (
        qf * gammas[1] * sqrt13
    )  # (q*sqrt(1+q^2)/(2*eta))*(g2)*sqrt((-g1*g3)/(o1*o3)) = (q*g2/(2*eta))*sqrt((1+q^2)*(-g1*g3)/(o1*o3))
    jac[2, 0] = (
        -omegas[2] * sqrt13
    )  # -o3 * sqrt((-g1*g3)/(o1*o3)) = -sqrt(-g1*o3*g3/o1)
    jac[2, 1] = (
        -omegas[2] * sqrt23
    )  # -o3 * sqrt((g2*g3)/(o2*o3)) = -sqrt(g2*o3*g3/o2)
    jac[2, 3] = (
        qf * gammas[2] * sqrt12
    )  # (q*sqrt(1+q^2)/(2*eta))*(g3)*sqrt((-g1*g2)/(o1*o2)) = (q*g3/(2*eta))*sqrt((1+q^2)*(-g1*g2)/(o1*o2))
    jac[3, 0] = (
        qsq_over_qf * (gplus - (2.0 * gammas[0])) / sqrt23
    )  # ((2*eta*q/sqrt(1+q^2))/sqrt((g2*g3)/(o2*o3)))*(gplus - 2*g1)
    jac[3, 1] = (
        qsq_over_qf * (gplus - (2.0 * gammas[1])) / sqrt13
    )  # ((2*eta*q/sqrt(1+q^2))/sqrt((-g1*g3)/(o1*o3)))*(gplus - 2*g2)
    jac[3, 2] = (
        qsq_over_qf * (gplus - (2.0 * gammas[2])) / sqrt12
    )  # ((2*eta*q/sqrt(1+q^2))/sqrt((-g1*g2)/(o1*o2)))*(gplus - 2*g3)
    jac[3, 3] = gplus  # g1 + g2 + g3 = gplus
    # return the jacobian
    return jac


def jacobian_in_fp_real_three_ae_system(
    gammas: 'npt.NDArray[np.float64]',
    omegas: 'npt.NDArray[np.float64]',
    q: 'float | npt.NDArray[np.float64]',
    eta: 'float | npt.NDArray[np.float64] | list[float]',
) -> 'npt.NDArray[np.float64]':
    """Compute the jacobian matrix in the fixed point for a system of three real-valued AEs (i.e. harmonic AEs).

    Notes
    -----
    The input gamma and omega data should be proved in numpy arrays of size 3. (Mode triad format!)

    Parameters
    ----------
    gammas : npt.NDArray[np.float64]
        Linear driving/damping rates of the harmonic mode triad.
    omegas : npt.NDArray[np.float64]
        Linear angular frequencies of the harmonic mode triad.
    q : float | npt.NDArray[np.float64]
        Ratio of the angular frequency detuning over the summed driving/damping rate.
    eta : float | npt.NDArray[np.float64]
        Coupling coefficient.

    Returns
    -------
    jac : npt.NDArray[np.float64]
        The jacobian matrix in the fixed point.
    """
    if isinstance(eta, list):
        eta = np.array(eta)
    # initialize the jacobian array
    jac = np.zeros((3, 3), dtype=np.float64)
    # store intermediate values
    inv_quality = gammas / omegas  # gx / ox
    og = gammas * omegas  # gx * ox
    sqrt12 = np.sqrt(
        -2.0 * inv_quality[0] * inv_quality[1]
    )  # sqrt(-2*g1*g2/(o1*o2))
    qsqrt = q * np.sqrt(1.0 + (q**2.0))  # q*sqrt(1+q^2)
    deta = 2.0 * eta  # 2*eta
    gplus = gammas.sum()  # gplus
    qf = qsqrt / deta  # q*sqrt(1+q^2) / (2*eta)
    qsq_over_qf = q**2.0 / qf  # (2*eta*q) / sqrt(1+q^2)
    # fill up the jacobian elements
    jac[np.diag_indices(2)] = gammas[:-1]  # (g1,g2)
    jac[0, 1] = np.sqrt(-og[0] * inv_quality[1])  # sqrt(-g1*o1 * g2 / o2)
    jac[0, 2] = (
        qf * gammas[0] * inv_quality[1]
    )  # (q*sqrt(1+q^2)/(2*eta))*(g1*g2/o2)
    jac[1, 0] = -np.sqrt(
        (-2.0 * inv_quality[0] * og[1])
    )  # -sqrt((-2*g1*g2*o2)/(o1))
    jac[1, 2] = (
        gammas[1] * qf * sqrt12
    )  # g1 * (q*sqrt(1+q^2)/(2*eta)) * sqrt(-2*g1*g2/(o1*o2))
    jac[2, 0] = (
        qsq_over_qf * (gplus - (2.0 * gammas[0])) / inv_quality[1]
    )  # 2*q*sqrt(1+q^2)*(2*g2-g1)*(o2/g2)
    jac[2, 1] = qsq_over_qf * np.sqrt(
        -2.0 * og[0] / inv_quality[1]
    )  # 2*q*sqrt(1+q^2) * sqrt((-2*o1*g1*o2)/(og2))
    jac[2, 2] = gplus  # g1 + 2*g2 = gplus
    # return the jacobian
    return jac
