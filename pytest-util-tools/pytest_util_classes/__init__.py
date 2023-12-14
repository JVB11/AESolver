"""Initialization file for the python package containing utility classes that facilitate the writing of pytest tests.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# custom imports
from .enumeration_classes import EFPVs, EPPs, EVs
from .data_classes import IIFTs


# define version, author and auto-import packages
__version__ = '1.0.0'
__author__ = 'Jordan Van Beeck'
__all__ = ['EFPVs', 'EPPs', 'EVs', 'IIFTs']
