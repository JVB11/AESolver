"""Initialization file for module containing functions that generate printouts of useful information about profiles.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# custom imports
from .coupling_coefficient_print_functions import (
    print_normed_cc_and_get_adjust_ratio,
    print_check_hyperbolic,
    print_surface_luminosities,
    print_theoretical_amplitudes,
    display_contributions,
)
from .coupling_coefficient_contributions_print_functions import (
    print_cc_contribution_info,
)


# define version, author and auto-import packages
__version__ = '1.0.0'
__author__ = 'Jordan Van Beeck'
__all__ = [
    'print_normed_cc_and_get_adjust_ratio',
    'print_check_hyperbolic',
    'print_surface_luminosities',
    'print_theoretical_amplitudes',
    'display_contributions',
    'print_cc_contribution_info',
]
