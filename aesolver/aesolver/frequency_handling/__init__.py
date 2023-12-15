"""Initialization file for the python package that handles frequency conversions from GYRE output.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# custom imports
from .frequency_handler import FrequencyHandler as FH


# define version, author and auto-import packages
__version__ = '1.0.0'
__author__ = 'Jordan Van Beeck'
__all__ = ['FH']
