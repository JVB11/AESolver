"""Python initialization file for the package that verifies the stability of the two-mode (harmonic) stationary solutions of the amplitude equations.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# custom imports
from .stab_checker_quadratic_two_sum import (
    check_dziembowski_conditions_direct as check_dziembowski_direct,
)
from .stab_checker_quadratic_two_sum import (
    check_dziembowski_conditions_parametric as check_dziembowski_parametric,
)


# define version, author and auto-import packages
__version__ = '1.0.0'
__author__ = 'Jordan Van Beeck'
__all__ = ['check_dziembowski_direct', 'check_dziembowski_parametric']
