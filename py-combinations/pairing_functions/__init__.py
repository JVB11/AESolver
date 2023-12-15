"""Initialization file for the pairing functions module.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""

# define version
__version__ = '1.0.0'


# define author
__author__ = 'Jordan Van Beeck'


# import functions to make them retrievable
from .cantor_pairing import cantor_pair, invert_cantor_pair
from .symmetric_pairing import symmetric_pair, invert_symmetric_pair
from .xie_pairing import xie_symmetric_pair, invert_xie_symmetric_pair


# define all package functions that can possibly be imported
__all__ = [
    'cantor_pair',
    'invert_cantor_pair',
    'symmetric_pair',
    'invert_symmetric_pair',
    'xie_symmetric_pair',
    'invert_xie_symmetric_pair',
]
