"""Initialization file for Python module that contains classes that aid in plotting profiles of the coupling coefficient and its contributions.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# custom imports
from .plot_template import (
    double_masking_integrated_radial_profile_figure_template_function,
    single_integrated_radial_profile_figure_template_function,
)


# define version, author and auto-import packages
__version__ = '1.0.0'
__author__ = 'Jordan Van Beeck'
__all__ = [
    'double_masking_integrated_radial_profile_figure_template_function',
    'single_integrated_radial_profile_figure_template_function',
]
