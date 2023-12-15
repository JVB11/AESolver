"""Python initialization file for the data handling component of the coupling package that computes nonlinear mode coupling coefficients.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# custom imports
from .storer import NumericalSaver
from .unpacker import UnPacker


# define version, author and auto-import packages
__version__ = '1.0.0'
__author__ = 'Jordan Van Beeck'
__all__ = ['NumericalSaver', 'UnPacker']
