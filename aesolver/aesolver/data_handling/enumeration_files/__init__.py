"""Python initialization file for the submodule containing enumeration modules for storage of numerical data from nonlinear mode coupling simulations/calculations.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# custom imports
from .enum_generic import SaveDirectoryEnum as SaveDirEnum
from .enum_generic import SaveSubDirectory as SaveSubDir
from .enum_hdf5_saver import Hdf5FileOpenModeSelector as Hdf5Open


# define version, author and auto-import packages
__version__ = '1.0.0'
__author__ = 'Jordan Van Beeck'
__all__ = ['SaveDirEnum', 'SaveSubDir', 'Hdf5Open']
