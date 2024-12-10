"""Initialization file for the python module that contains a class used for loading data from inlists into Python.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
__version__ = '1.0.0'
__author__ = 'Jordan Van Beeck'


from .inlist_handler import InlistHandler
from .toml_inlist_handler import TomlInlistHandler


# define all modules to be imported for 'import *'
__all__ = ['InlistHandler', 'TomlInlistHandler']
