"""Initialization file for the Python package that contains functionalities/classes to perform angular integrations necessary to compute the coupling coefficients for rotating stars, within the TAR.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# custom imports
from .hough_function_handler import HoughFunctionHandler as HFHandler
from .hough_integration import HoughIntegration as HI


# define version, author and auto-import packages
__version__ = '1.0.0'
__author__ = 'Jordan Van Beeck'
__all__ = ['HFHandler', 'HI']
