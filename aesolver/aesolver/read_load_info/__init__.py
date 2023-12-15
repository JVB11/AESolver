"""Initialization file for the read and load modules of the AEsolver package.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# custom import
from .read_info_dirs import InformationDirectoriesHandler as InfoDirHandler
from .load_name_plots import generate_load_file_path_list


# define version, author and auto-import packages
__version__ = '1.0.0'
__author__ = 'Jordan Van Beeck'
__all__ = ['InfoDirHandler', 'generate_load_file_path_list']
