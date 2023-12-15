"""Python initialization file for the submodule containing file-format specific saving methods for the data generated from nonlinear mode coupling simulations/calculations.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# custom import
from .hdf5_saver import HDF5Saver


# define version, author and auto-import packages
__version__ = '1.0.0'
__author__ = 'Jordan Van Beeck'
__all__ = ['HDF5Saver']
