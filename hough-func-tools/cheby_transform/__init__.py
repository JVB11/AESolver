"""Initialization file for the python module that contains a class to perform Chebyshev collocation.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""

# version + author
__version__ = '1.0.0'
__author__ = 'Jordan Van Beeck'


# import the collocation class
from .cheby_transform import ChebyTransform


# make ChebyTransform available for import
__all__ = ['ChebyTransform']
