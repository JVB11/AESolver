"""Initialization file for the Python module containing functions that compute combination lists of inputs and their cardinalities.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""

# define version
__version__ = '1.0.0'


# define author
__author__ = 'Jordan Van Beeck'


# import classes defined in individual modules
from .symmetric_product import SymmetricProductCombos
from .single_set_combinations import SingleSetCombos
from .cartesian_combinations import CartesianCombos
from .generic_combination_class import GenericCombination


# define all package classes that can possibly be imported
__all__ = [
    'SymmetricProductCombos',
    'SingleSetCombos',
    'CartesianCombos',
    'GenericCombination',
]
