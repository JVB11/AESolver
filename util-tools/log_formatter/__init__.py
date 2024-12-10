"""Initialization file for the python module that contains functions used to set up the root logger, so that it logs messages in a nice format.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
__version__ = '1.0.0'
__author__ = 'Jordan Van Beeck'


from .log_formatter import set_up_root_logger
from .log_formatter import adjust_root_logger_level


# define all modules to be imported for 'import *'
__all__ = ['set_up_root_logger', 'adjust_root_logger_level']
