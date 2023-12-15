"""Initialization file of the sub-package containing helper functions for the saving of HDF5 files.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# custom imports
from .create_group import create_group_path_dataset
from .overwrite_dataset import overwrite_dataset


# define version, author and auto-import packages
__version__ = '1.0.0'
__author__ = 'Jordan Van Beeck'
__all__ = ['create_group_path_dataset', 'overwrite_dataset']
