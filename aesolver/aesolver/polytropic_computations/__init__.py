"""Initialization file for the module that performs computations that provide additional data arrays necessary for computing coupling coefficients for g modes in polytropes.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# custom imports
from .poly_struct_comps import PolytropicAdditionalStructureInformation as PaSi


# define version, author and auto-import packages
__version__ = '1.0.0'
__author__ = 'Jordan Van Beeck'
__all__ = ['PaSi']
