"""Initialization file for the python package that handles MESA & GYRE model data.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# custom imports
from .load_models import ModelLoader


# define version, author and auto-import packages
__version__ = '1.0.0'
__author__ = 'Jordan Van Beeck'
__all__ = ['ModelLoader']
