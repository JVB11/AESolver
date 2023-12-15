"""Initialization file for Python module that contains classes that aid in plotting profiles of radial and angular kernels of integrals appearing in the coupling coefficient.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# custom imports
from .angular_kernel_plots import (
    create_angular_contribution_figures_for_specific_contribution,
)
from .radial_kernel_plots import (
    create_radial_contribution_figures_for_specific_contribution,
)


# define version, author and auto-import packages
__version__ = '1.0.0'
__author__ = 'Jordan Van Beeck'
__all__ = [
    'create_angular_contribution_figures_for_specific_contribution',
    'create_radial_contribution_figures_for_specific_contribution',
]
