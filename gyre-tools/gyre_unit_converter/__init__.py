"""Initialization file for the python module that contains classes to perform unit conversion for GYRE output data.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import the UnitConverter classes
from .gyre_unit_converter import GYREFreqUnitConverter


# version + author
__version__ = '1.0.0'
__author__ = 'Jordan Van Beeck'
__all__ = ['GYREFreqUnitConverter']
