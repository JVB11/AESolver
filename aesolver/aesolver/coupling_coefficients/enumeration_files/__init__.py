"""Initialization file for Python subpackage containing helpful enumeration files.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# custom imports
from .enumerated_quadratic_coupling import EnumRadial, EnumRadialPoly
from .enumeration_mode_data import ModeData


# define version, author and auto-import packages
__version__ = '1.0.0'
__author__ = 'Jordan Van Beeck'
__all__ = ['EnumRadial', 'EnumRadialPoly', 'ModeData']
