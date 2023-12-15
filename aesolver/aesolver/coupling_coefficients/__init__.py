"""Initialization file for the python package that handles computation of coupling coefficients.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# custom imports
from .quadratic_coupling_coefficient_rotating import (
    QuadraticCouplingCoefficientRotating as QCCR,
)


# define version, author and auto-import packages
__version__ = '1.0.0'
__author__ = 'Jordan Van Beeck'
__all__ = ['QCCR']
