"""Python initialization file for the package that verifies the stability of the three-mode stationary solutions of the amplitude equations.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# custom imports
from .stab_checker_quadratic_three_sum import gamma_sum_stability as gss
from .stab_checker_quadratic_three_sum import (
    check_corrected_dziembowski_conditions as check_van_beeck_conditions,
)


# define version, author and auto-import packages
__version__ = '1.0.0'
__author__ = 'Jordan Van Beeck'
__all__ = ['gss', 'check_van_beeck_conditions']
