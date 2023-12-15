'''Module containing utility functions for the plotter class.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
'''
# import statements
import numpy as np


# FREQUENCY-RELATED QUANTITY CONVERSION FUNCTIONS
def hz_to_cpd(freqs_in_hz: float | np.ndarray) -> float | np.ndarray:
    return freqs_in_hz * 86400.0


def hz_to_muhz(freqs_in_hz: float | np.ndarray) -> float | np.ndarray:
    return freqs_in_hz * 1.0e6


# COMPUTE SINGLE AMPLITUDE RATIO
def compute_single_amplitude_ratio(
    surf_numerator: np.ndarray,
    surf_denominator: np.ndarray,
    specified_out: np.ndarray,
    additional_mask: np.ndarray,
) -> None:
    denom_mask = (surf_denominator == 0.0) | np.isnan(surf_denominator)
    np.divide(
        surf_numerator,
        surf_denominator,
        out=specified_out,
        where=np.logical_and(~denom_mask, additional_mask),
    )
