"""Initialization file for the package containing a class that computes the stationary solutions for the AEs for three-mode coupling (see e.g. Van Beeck et al. 2022)

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# custom imports
from .three_mode_stationary import ThreeModeStationary


# define version, author and auto-import packages
__version__ = '1.0.0'
__author__ = 'Jordan Van Beeck'
__all__ = ['ThreeModeStationary']
