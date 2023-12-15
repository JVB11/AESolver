"""Python initialization file for the package containing sub-packages that verify the stability of the stationary solutions of the amplitude equations.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# custom imports
from .hyperbolicity import check_hyper, hyper_jac, hyper_jac_three
from .three_mode import gss, check_van_beeck_conditions
from .two_mode import check_dziembowski_direct, check_dziembowski_parametric


# define version, author and auto-import packages
__version__ = '1.0.0'
__author__ = 'Jordan Van Beeck'
__all__ = [
    'check_hyper',
    'hyper_jac',
    'hyper_jac_three',
    'gss',
    'check_van_beeck_conditions',
    'check_dziembowski_direct',
    'check_dziembowski_parametric',
]
