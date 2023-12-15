"""Initialization file for the Python package that handles stationary solutions of the coupled-mode and amplitude equations.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# custom imports
from .generic import StatSolve
from .three_mode import ThreeModeStationary


# define version, author and auto-import packages
__version__ = '1.0.0'
__all__ = ['StatSolve', 'ThreeModeStationary']
__author__ = 'Jordan Van Beeck'
