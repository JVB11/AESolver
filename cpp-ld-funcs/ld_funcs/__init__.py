"""Initialization file for the ld_funcs library.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# extra information on the package
__version__ = '1.0.0'
__author__ = 'Jordan Van Beeck'


# import statement for module
from . import libcpp_ld_funcs as ldf  # type: ignore


__all__ = ['ldf']
