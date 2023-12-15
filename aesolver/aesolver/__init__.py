'''Initialization file for the installable AEsolver package. For additional information on this package, see https://github.com/JVB11/amplitude_equations_solver

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
'''
# custom imports
from .solver import QuadRotAESolver, QuadraticAEGridSolver, QuadraticAEProfileSolver
from .data_handling import NumericalSaver
from .ad_nad_comparer import AdNadComparer
from .mode_input_generator import InputGen
from .read_load_info import generate_load_file_path_list

# define version, author and auto-import packages
__version__ = '1.0.0'
__author__ = 'Jordan Van Beeck'
__all__ = ['AdNadComparer', 'InputGen', 'QuadRotAESolver', 'NumericalSaver', 'generate_load_file_path_list', 'QuadraticAEGridSolver', 'QuadraticAEProfileSolver']
