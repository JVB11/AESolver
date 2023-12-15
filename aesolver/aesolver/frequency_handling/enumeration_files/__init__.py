"""Initialization file for Python subpackage containing helpful enumeration files.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# custom imports
from .enumeration_freq_converter_normalization import is_angular_frequency
from .enumeration_freq_converter_normalization import (
    EnumerationGYREFreqConverter as EnumGYREFreqConv,
)


# define version, author and auto-import package
__version__ = '1.0.0'
__author__ = 'Jordan Van Beeck'
__all__ = ['EnumGYREFreqConv', 'is_angular_frequency']
