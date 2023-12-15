"""Initialization file for the python package that handles specific pre-checks for coupling coefficients.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# custom imports
from .three_mode_checks import check_direct, check_parametric, check_driven
from .three_mode_checks import PreCheckThreeModesQuadratic as PreThreeQuad


# define version, author and auto-import packages
__version__ = '1.0.0'
__author__ = 'Jordan Van Beeck'
__all__ = ['check_direct', 'check_parametric', 'check_driven', 'PreThreeQuad']
