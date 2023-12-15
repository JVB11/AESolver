"""Initialization python file for the package containing modules/classes to compare adiabatic and non-adiabatic results of stellar pulsation code computations (GYRE).

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# custom import
from .comparer import AdNadComparer


# define version, author and auto-import packages
__version__ = '1.0.0'
__author__ = 'Jordan Van Beeck'
__all__ = ['AdNadComparer']
