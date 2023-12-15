"""Initialization file for the python package that handles pre-checks for coupling coefficients.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# custom imports
from .specific import PreThreeQuad, check_direct, check_parametric, check_driven
from .generic import PreCheckQuadratic


# define version, author and auto-import packages
__version__ = '1.0.0'
__author__ = 'Jordan Van Beeck'
__all__ = [
    'PreThreeQuad',
    'check_direct',
    'check_driven',
    'check_parametric',
    'PreCheckQuadratic',
]
