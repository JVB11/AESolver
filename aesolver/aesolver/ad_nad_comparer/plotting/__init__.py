"""Initialization python file for the sub-package containing modules/classes to generate figures that compare adiabatic and non-adiabatic results of stellar pulsation code computations (GYRE).

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# custom imports
from .gyre_plotter import AdNadComparisonGYREPlotter as GYREPlot
from .mesa_plotter import AdNadComparisonMESAPlotter as MESAPlot


# define version, author and auto-import packages
__version__ = '1.0.0'
__author__ = 'Jordan Van Beeck'
__all__ = ['GYREPlot', 'MESAPlot']
