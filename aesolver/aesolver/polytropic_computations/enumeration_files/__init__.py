"""Initialization file for the enumeration_files submodule of aesolver.polytropic_computations.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# custom imports
from .polytropic_struct_enumeration import PolytropicDerivativeConversion as PdC
from .polytropic_struct_enumeration import PolytropicStructureOutput as PSo

# define version, author and auto-import packages
__version__ = '1.0.0'
__author__ = 'Jordan Van Beeck'
__all__ = ['PdC', 'PSo']
