"""Python initialization file for the package containing dataclasses that facilitate the analysis of the amplitude equation solutions.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import modules
from .hdf5_data_path_mapper import HDF5MappingsAttributes, HDF5MappingsArrays


__version__ = '1.0.0'
__all__ = [
    'HDF5MappingsAttributes',
    'HDF5MappingsArrays',
    'OverviewPlotOptions',
]
__author__ = 'Jordan Van Beeck'
