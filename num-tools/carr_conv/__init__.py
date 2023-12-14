"""Initialization file for the python module that contains functions that can be used for array conversions of Numpy arrays loaded from GYRE data.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import the functions in the module
from .carr_conv import re_im
from .carr_conv import re_im_parts


# version + author + functions
__version__ = '1.0.0'
__author__ = 'Jordan Van Beeck'
__all__ = ['re_im', 're_im_parts']
