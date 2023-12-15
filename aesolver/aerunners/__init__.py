'''Python package containing run scripts that can be used to start AESolver runs.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
'''
# import runner modules
from .solve_quadratic import perform_complete_solve_run
from .plot_quadratic import perform_complete_plotting_run
from .analyze_quadratic import perform_complete_analysis_run


# define version, author and auto-import packages
__version__ = '1.0.0'
__author__ = 'Jordan Van Beeck'
__all__ = ['perform_complete_solve_run', 'perform_complete_plotting_run', 'perform_complete_analysis_run']
