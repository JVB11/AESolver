"""Python initialization file for the shared functionality sub-module.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import shared functions
from .shared_check_functions import check_data_shape, check_input_data
from .shared_computation_functions import (
    compute_bin_width,
    get_parameter_combinations,
    get_peak_to_peak_range,
    histogram_log_counts_and_edges,
    histogram_log_counts,
    empirical_probability_histogram_log_counts,
    empirical_density_histogram_log_counts,
)
from .shared_handling_functions import (
    handle_single_data_point,
    handle_return_binning_parameters,
)
from .shared_log_functions import print_maximal_values, print_single_value
from .shared_masking_functions import retrieve_valued_data


# make extra information available
__version__ = '1.0.0'
__author__ = 'Jordan Van Beeck'
__all__ = [
    'check_data_shape',
    'check_input_data',
    'compute_bin_width',
    'get_parameter_combinations',
    'get_peak_to_peak_range',
    'histogram_log_counts_and_edges',
    'histogram_log_counts',
    'empirical_probability_histogram_log_counts',
    'empirical_density_histogram_log_counts',
    'handle_single_data_point',
    'print_maximal_values',
    'print_single_value',
    'retrieve_valued_data',
    'handle_return_binning_parameters',
]
