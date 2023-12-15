"""Initialization file for Python module that contains classes that aid in plotting profiles of quantities such as the coupling coefficient, its contributions, and its radial and angular kernels.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# custom imports
from .setup_for_plotting import (
    get_figure_output_base_path,
    get_model_dependent_part_save_name,
    save_fig,
)
from .coupling_coefficient_plots import (
    double_masking_integrated_radial_profile_figure_template_function,
    single_integrated_radial_profile_figure_template_function,
)
from .kernel_plots import (
    create_angular_contribution_figures_for_specific_contribution,
    create_radial_contribution_figures_for_specific_contribution,
)


# define version, author and auto-import packages
__version__ = '1.0.0'
__author__ = 'Jordan Van Beeck'
__all__ = [
    'get_figure_output_base_path',
    'get_model_dependent_part_save_name',
    'save_fig',
    'double_masking_integrated_radial_profile_figure_template_function',
    'single_integrated_radial_profile_figure_template_function',
    'create_angular_contribution_figures_for_specific_contribution',
    'create_radial_contribution_figures_for_specific_contribution',
]
