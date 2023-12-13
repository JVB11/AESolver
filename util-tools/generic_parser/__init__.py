"""Python package initialization file for package containing a generic class that can be subclassed to provide a parser used to parse arguments from the command line and/or an inlist.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""

# version + author
__version__ = '1.0.0'
__author__ = 'Jordan Van Beeck'


# import the GenericParser object
from .generic_parser import GenericParser


# define all modules to be imported for 'import *'
__all__ = ['GenericParser']
