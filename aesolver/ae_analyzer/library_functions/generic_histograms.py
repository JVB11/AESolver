"""Module containing functions used to generate histograms for a variety of purposes.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import numpy as np

# custom package imports --> used elsewhere
from hogg_jacky import (
    histogram_likelihood_blas,
    log_histogram_likelihood_blas,
    histogram_blas_log_spec_edges,
)

# type checking imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    histogram_likelihood_blas
    log_histogram_likelihood_blas
    histogram_blas_log_spec_edges


def create_setup_histogram_likelihood_arrays(
    masked_data, max_nr_bins=100
) -> tuple[np.ndarray, np.ndarray]:
    my_bin_numbers = np.arange(
        2,
        min(100 if (max_nr_bins is None) else max_nr_bins, masked_data.shape[0])
        + 1,
    )
    alphas = np.power(10.0, np.arange(-2.0, 3.0))
    bin_phases = np.array([0.0], dtype=np.float64)
    return my_bin_numbers, alphas, bin_phases
