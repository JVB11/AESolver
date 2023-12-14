"""Initialization file for the 'pytest_util_functions' module, which contains utility functions to be used for writing pytests.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# custom imports
from .assertion_checker import check_assertions
from .comparison import value_comparison, compare_elements, compare_dicts


# define version, author and auto-import packages
__version__ = '1.0.0'
__author__ = 'Jordan Van Beeck'
__all__ = [
    'check_assertions',
    'value_comparison',
    'compare_elements',
    'compare_dicts',
]
