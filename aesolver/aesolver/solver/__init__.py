"""Python initialization file for the package containing the class that will handle the solving of the amplitude equations.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# custom imports
from .quadratic_solver import QuadraticAESolver as QuadRotAESolver
from .quadratic_grid import QuadraticAEGridSolver
from .quadratic_profile import QuadraticAEProfileSolver


# define version, author and auto-import packages
__version__ = '1.0.0'
__author__ = 'Jordan Van Beeck'
__all__ = ['QuadRotAESolver', 'QuadraticAEGridSolver', 'QuadraticAEProfileSolver']
