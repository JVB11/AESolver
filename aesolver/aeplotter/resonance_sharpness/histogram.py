"""Module containing function to compute the sharpness histogram.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import sys
import numpy as np
import logging

# imports from other submodules
from ..library_functions import (
    create_setup_histogram_likelihood_arrays,
    histogram_likelihood_blas,
    log_histogram_likelihood_blas,
    histogram_blas_log_spec_edges,
)

# type checking imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any
    import numpy.typing as npt
    from .figure_class import ResonanceSharpnessFigure


logger = logging.getLogger(__name__)


def _sharpness_histogram(
    my_sharpness_data: 'npt.NDArray[np.float64]',
    log_scale: bool = False,
    max_nr_bins: int = 100,
    empirical_density: bool = False,
    empirical_probability: bool = True,
) -> 'tuple[npt.NDArray[np.float64] | None, npt.NDArray[np.float64] | None, np.float64 | None, Any]':
    # create setup info
    my_bin_nrs, alphas, bin_phases = create_setup_histogram_likelihood_arrays(
        masked_data=my_sharpness_data, max_nr_bins=max_nr_bins
    )
    # compute histogram information
    if empirical_probability != empirical_density:
        if log_scale:
            (
                empirical_histogram_data,
                hist_edges,
                bin_width,
                binning_params,
            ) = log_histogram_likelihood_blas(
                my_data=my_sharpness_data,
                my_alpha_parameters=alphas,
                my_bin_numbers=my_bin_nrs,
                my_bin_phases=bin_phases,
                return_binning_parameters=True,
                empirical_density=empirical_density,
                empirical_probability=empirical_probability,
            )
        else:
            (
                empirical_histogram_data,
                hist_edges,
                bin_width,
                binning_params,
            ) = histogram_likelihood_blas(
                my_data=my_sharpness_data,
                my_alpha_parameters=alphas,
                my_bin_numbers=my_bin_nrs,
                my_bin_phases=bin_phases,
                return_binning_parameters=True,
                empirical_density=empirical_density,
                empirical_probability=empirical_probability,
            )
        # return histogram info
        return empirical_histogram_data, hist_edges, bin_width, binning_params
    else:
        logger.error(
            'Asking for wrong input in the _sharpness_histogram function! Now exiting.'
        )
        sys.exit()


def get_sharpness_histogram_plot_data(
    resonance_sharpness_object: 'ResonanceSharpnessFigure'
) -> None:
    # compute the sharpness histogram
    _, log_hist_edges_all, log_bin_width_all, _ = _sharpness_histogram(
        my_sharpness_data=np.abs(resonance_sharpness_object.sharpness_all),
        log_scale=True,
        empirical_probability=True,
        max_nr_bins=100,
        empirical_density=False,
    )
    # get the minimal number of bins (GUARD)
    my_min_nr_bins = 1 if log_bin_width_all is None else len(log_bin_width_all)
    # compute sharpness histogram for isolated triads
    log_hist_prob_isolated_all = histogram_blas_log_spec_edges(
        my_data=np.abs(resonance_sharpness_object.sharpness_isolated),
        my_bin_edges=log_hist_edges_all,
        empirical_probability=True,
        empirical_density=False,
        min_nr_bins=my_min_nr_bins,
    )
    # compute sharpness histogram for non-isolated triads
    log_hist_prob_not_isolated_all = histogram_blas_log_spec_edges(
        my_data=np.abs(resonance_sharpness_object.sharpness_not_isolated),
        my_bin_edges=log_hist_edges_all,
        empirical_probability=True,
        empirical_density=False,
        min_nr_bins=my_min_nr_bins,
    )
    # store the resonance sharpness plotting information
    resonance_sharpness_object.edges = log_hist_edges_all
    resonance_sharpness_object.bin_width = log_bin_width_all
    resonance_sharpness_object.prob_isolated_log = log_hist_prob_isolated_all
    resonance_sharpness_object.prob_not_isolated_log = (
        log_hist_prob_not_isolated_all
    )
