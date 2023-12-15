"""Initialization file for the enumeration files of the load/read modules of the AE solver code.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# custom import names
from .enumeration_read import InfoDictConverter as InfoDictConv
from .enumeration_read import EnumerationInitializationDirs as EnumInitDir
from .enumeration_read import EnumerationLoadLists as EnumLoadLists


# define version, author and auto-import packages
__version__ = '1.0.0'
__author__ = 'Jordan Van Beeck'
__all__ = ['InfoDictConv', 'EnumInitDir', 'EnumLoadLists']
