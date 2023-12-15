"""Python module containing functions that print information about the coupling coefficient profiles.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import numpy as np

# intra-package imports
from ...stability_checker import check_hyper, hyper_jac, hyper_jac_three

# type checking imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..quadratic_solver import QuadraticAESolver
    from ...frequency_handling import FH


def print_normed_cc_and_get_adjust_ratio(cc_val: float | np.float64, adjusted_cc_val: float | np.float64) -> float | np.float64:
    # add some space before printouts
    print('\n\n')
    # compute the adjusted quantities and print the results
    print(
        f'Normed coupling coefficient eta: {cc_val}, after adjustment for TAR validity: {adjusted_cc_val}'
    )
    # compute and return the adjustment ratio
    return adjusted_cc_val / cc_val


def print_theoretical_amplitudes(
    stat_sol_info: dict, adjust_ratio: float
) -> np.ndarray:
    # display the equilibrium amplitudes
    stat_amps = stat_sol_info['equilibrium amplitudes'] * 1.0e6
    stat_amps_adjust = stat_amps / adjust_ratio
    print(
        f'Stationary amplitudes (ppm): {stat_amps}, after adjustment for TAR validity (ppm): {stat_amps_adjust}'
    )
    # display the threshold amplitudes
    _threshold_amp = stat_sol_info['critical parent amplitude'] * 1.0e6
    _threshold_amp_adjust = _threshold_amp / adjust_ratio
    print(
        f'Parent threshold amplitude for parametric resonance (ppm): {_threshold_amp}, after adjustment for TAR validity (ppm): {_threshold_amp_adjust}'
    )
    # return the stationary amplitudes used to compute a conversion factor
    return stat_amps


def print_surface_luminosities(
    stat_sol_info: dict, adjust_ratio: float, stat_amps: np.ndarray
) -> None:
    # display the mode luminosity fluctuations at the surface
    _surface_fluxes = stat_sol_info['surface flux variations'] * 1.0e6
    _surface_fluxes_adjust = _surface_fluxes / adjust_ratio
    print(
        f'Surface luminosity fluctuations (ppm): {_surface_fluxes}, after adjustment for TAR validity (ppm): {_surface_fluxes_adjust}'
    )
    # compute the parent threshold surface luminosities and display them
    _amp_to_flux = _surface_fluxes / stat_amps  # CONVERSION FACTOR
    _threshold_flux = (
        stat_sol_info['critical parent amplitude'] * _amp_to_flux[0] * 1.0e6
    )
    _threshold_flux_adjust = _threshold_flux / adjust_ratio
    print(
        f'Parent threshold luminosity fluctuation for parametric resonance (ppm): {_threshold_flux}, after adjustment for TAR validity (ppm): {_threshold_flux_adjust}'
    )


def display_contributions(
    contribution_array_masked: list[float] | list[list[float]]
) -> None:
    # display the individual contributions due to TAR non-validity regions
    print(
        f'Contributions to the normed coupling coefficient due to TAR non-validity zones: {contribution_array_masked}. These contributions are ordered in terms of the values of r/R for which the TAR validity criterion start to become invalid: the first number for example is the contribution of the TAR invalidity region in the core region.'
    )


def print_check_hyperbolic(
    solving_object: 'QuadraticAESolver',
    freq_handler: 'FH',
    adjusted_cc: float | list[float],
):
    # CHECK HYPERBOLICITY WITH THE CHANGED COUPLING COEFFICIENT
    if TYPE_CHECKING:
        assert solving_object._gyre_ad
    two_mode_resonance = solving_object.check_two_mode_sum_resonance(
        solving_object._gyre_ad
    )
    if two_mode_resonance:
        _my_adjusted_jac = hyper_jac_three(
            freq_handler.driving_rates,
            freq_handler.corot_mode_omegas,
            freq_handler.q_factor,
            eta=adjusted_cc,
        )
    else:
        _my_adjusted_jac = hyper_jac(
            freq_handler.driving_rates,
            freq_handler.corot_mode_omegas,
            freq_handler.q_factor,
            eta=adjusted_cc,
        )
    _adjusted_check_hyper = check_hyper(my_jac=_my_adjusted_jac)
    print(
        f'Hyperbolicity check: {solving_object._hyperbolicity_info[0]}, after adjustment for TAR validity: {_adjusted_check_hyper}'
    )
    # add space after prints
    print('\n\n')
