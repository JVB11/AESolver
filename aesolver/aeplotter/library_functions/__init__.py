"""Initialization file for the plotting utility module containing library functions used to generate plots.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import library functions
from .masking import (
    isolated_non_isolated_stable_valid_masking,
    theoretical_plotting_masks_overview,
)
from .generic_histograms import (
    histogram_likelihood_blas,
    log_histogram_likelihood_blas,
    histogram_blas_log_spec_edges,
    create_setup_histogram_likelihood_arrays,
)


__all__ = [
    'isolated_non_isolated_stable_valid_masking',
    'histogram_likelihood_blas',
    'log_histogram_likelihood_blas',
    'histogram_blas_log_spec_edges',
    'create_setup_histogram_likelihood_arrays',
    'theoretical_plotting_masks_overview',
]
__version__ = '1.0.0'
__author__ = 'Jordan Van Beeck'
