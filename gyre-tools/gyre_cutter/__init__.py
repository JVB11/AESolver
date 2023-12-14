"""Initialization file for the python module that contains class used to cut boundary points of the GYRE eigenfunctions.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import the GYREEigenFunctionNormalizer class
from .gyre_cutter import EnumerationCutter as GYRECutter


# version + author
__version__ = '1.0.0'
__author__ = 'Jordan Van Beeck'
__all__ = ['GYRECutter']
