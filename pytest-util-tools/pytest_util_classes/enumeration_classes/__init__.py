"""Initialization file for the submodule 'enumeration_classes' which contains the enumeration classes for the 'pytest_util_classes' module.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# custom imports
from .enumerate_failing_params import EnumerateFailingParameterValues as EFPVs
from .enumerate_pytest_params import EnumeratePytestParameters as EPPs
from .enumerate_values import EnumeratedValues as EVs


# define version, author and auto-import packages
__version__ = '1.0.0'
__author__ = 'Jordan Van Beeck'
__all__ = ['EFPVs', 'EPPs', 'EVs']
