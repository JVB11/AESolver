"""Python module containing functions shared among the different jackknife likelihood computing functionalities.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import modules
import logging
import numpy as np
from itertools import product

# import type aliases from separate module
from ..typing_info import (
    t_arr,
    D_var,
    D_arr,
    D_list,
    I_arr,
    I_list,
    I_var,
    Iter_var,
    Iter_var3,
)


# set up a logger
logger = logging.getLogger(__name__)


def get_peak_to_peak_range(my_data: t_arr) -> D_var:
    """Computes the peak to peak range of the data array.

    Parameters
    ----------
    my_data : t_arr
        Data array for which jackknife likelihood is to be computed.

    Returns
    -------
    D_var
        Peak to peak data range.
    """
    return np.ptp(my_data)


def get_parameter_combinations(
    my_alpha_parameters: D_arr | D_list,
    my_bin_numbers: I_arr | I_list,
    my_bin_phases: D_arr | D_list | None = None,
    return_list: bool = True,
) -> (
    list[tuple[D_var, I_var, D_var]]
    | Iter_var3
    | list[tuple[D_var, I_var]]
    | Iter_var
):
    """Retrieves the list of parameter combinations, as computed
    by the itertools module.

    Parameters
    ----------
    my_alpha_parameters : D_arr | D_list
        Contains the alpha smoothing parameters for the Hogg(2008) jackknife likelihood.
    my_bin_numbers : I_arr | I_list
        Contains several numbers of bins to be tried when computing the jackknife likelihood.
    return_list : bool, optional
        Returns the list of combinations (True) or return the iterator (False); by default True.

    Returns
    -------
    my_iter : list[tuple[D_var, I_var, D_var]] | Iter_var3 | list[tuple[D_var, I_var]] | Iter_var
        Combinations over all the available parameters (for the options that use numpy histogram to perform binning, bin phases are not used!).
    """
    # generate the iterator
    if my_bin_phases is None:
        my_iter = product(my_alpha_parameters, my_bin_numbers)
    else:
        my_iter = product(my_alpha_parameters, my_bin_numbers, my_bin_phases)
    # return a list or the iterator itself
    return list(my_iter) if return_list else my_iter


def compute_bin_width(my_ptp_range: D_var, my_bin_number: I_var) -> D_var:
    """Computes the bin width based on the peak-to-peak data range.

    Parameters
    ----------
    my_ptp_range : D_var
        Peak-to-peak data range.
    my_bin_number : I_var
        Number of bins to divide into.

    Returns
    -------
    D_var
        Bin width.
    """
    return my_ptp_range / float(my_bin_number)


# LOGARITHMIC BINNING FUNCTIONS


def histogram_log_counts_and_edges(
    my_log_data: D_arr,
    my_data: D_arr,
    bin_nr: I_var,
    bin_phase: D_var = 0.0,
    min_nr_bins: I_var = 0,
) -> tuple[D_arr, D_arr]:
    """Binning and bin edge-searching function for histograms with bin widths that are constant in log-space.

    Parameters
    ----------
    my_log_data : D_arr
        Log10 of data array for which jackknife likelihood is to be computed.
    my_data : D_arr
        Data array for which jackknife likelihood is to be computed.
    bin_nr : I_var
        The nr. of bins in the histogram.
    bin_phase : D_var, optional
        Binning phase; by default 0.0.
    min_nr_bins : I_var, optional
        The minimum number of bins to be considered; by default 0.

    Returns
    -------
    tuple[D_arr, D_arr]
        Histogram counts and bin edges.
    """
    # compute logarithmic histogram bin edges
    if bin_nr == 1:
        # ONE BIN: COVERS ALL DATA
        return np.array(
            [my_log_data.shape[0]], dtype=my_log_data.dtype
        ), np.array([my_data.min(), my_data.max()], dtype=my_log_data.dtype)
    else:
        # PERFORM BINNING
        bin_edges = np.logspace(
            my_log_data.min(),
            my_log_data.max(),
            num=bin_nr + 1,
            endpoint=True,
            dtype=my_log_data.dtype,
            base=10.0,
        )
        # return the histogram bin counts and edges
        return histogram_log_counts(
            my_data=my_data,
            my_bin_edges=bin_edges,
            binning_phase=bin_phase,
            min_nr_bins=min_nr_bins,
        ), bin_edges


def histogram_log_counts(
    my_data: D_arr,
    my_bin_edges: D_arr,
    binning_phase: D_var,
    min_nr_bins: I_var = 0,
) -> D_arr:
    """Binning procedure for histograms with bins that are equally wide in log-space.

    Parameters
    ----------
    my_data : D_arr
        Data array for which jackknife likelihood is to be computed.
    my_bin_edges : D_arr
        Edges determined by previously ran optimization.
    binning_phase : D_var, optional
        Binning phase; by default 0.0.
    min_nr_bins : I_var, optional
        The minimum number of bins to be considered; by default 0.

    Returns
    -------
    D_arr
        Histogram counts for the specified bin edges.
    """
    # compute bin_nr from shape of bin_edges
    bin_nr = my_bin_edges.shape[0] - 1
    # use search sorted operation to perform binning
    _binning_result = np.searchsorted(my_bin_edges, my_data, side='left') - 1
    # GUARDS:
    # adjust for negative values due to lower limit being reached.
    _binning_result[_binning_result == -1] = 0
    # adjust for bin_nr values due to upper limit being reached.
    _binning_result[_binning_result == bin_nr] = bin_nr - 1
    # log that you cannot use a constant binning phase for log-transformed data
    logger.debug(
        f'Cannot use a constant binning phase ({binning_phase}), when generating log-transformed data histograms (whose bins are not of equal width).'
    )
    # return the histogram bin counts and edges
    return np.bincount(_binning_result, minlength=min_nr_bins).astype(
        np.float64, copy=False
    )


def empirical_probability_histogram_log_counts(
    my_data: D_arr,
    my_bin_edges: D_arr,
    binning_phase: D_var,
    min_nr_bins: I_var = 0,
) -> D_arr:
    """Binning procedure for empirical probability histograms with bins that are equally wide in log-space.

    Parameters
    ----------
    my_data : D_arr
        Data array for which jackknife likelihood is to be computed.
    my_bin_edges : D_arr
        Edges determined by previously ran optimization.
    binning_phase : D_var, optional
        Binning phase; by default 0.0.
    min_nr_bins : I_var, optional
        The minimum number of bins to be considered; by default 0.

    Returns
    -------
    D_arr
        Histogram counts for the specified bin edges.
    """
    # return the histogram bin counts and edges
    return (
        histogram_log_counts(
            my_data=my_data,
            my_bin_edges=my_bin_edges,
            binning_phase=binning_phase,
            min_nr_bins=min_nr_bins,
        )
        / my_data.shape[0]
    )


def empirical_density_histogram_log_counts(
    my_data: D_arr,
    my_bin_edges: D_arr,
    binning_phase: D_var,
    min_nr_bins: I_var = 0,
) -> D_arr:
    """Binning procedure for empirical density histograms with bins that are equally wide in log-space.

    Parameters
    ----------
    my_data : D_arr
        Data array for which jackknife likelihood is to be computed.
    my_bin_edges : D_arr
        Edges determined by previously ran optimization.
    binning_phase : D_var, optional
        Binning phase; by default 0.0.
    min_nr_bins : I_var, optional
        The minimum number of bins to be considered; by default 0.

    Returns
    -------
    D_arr
        Histogram counts for the specified bin edges.
    """
    # compute the bin widths
    _bin_widths = my_bin_edges[1:] - my_bin_edges[:-1]
    # return the histogram bin counts and edges
    return (
        empirical_probability_histogram_log_counts(
            my_data=my_data,
            my_bin_edges=my_bin_edges,
            binning_phase=binning_phase,
            min_nr_bins=min_nr_bins,
        )
        / _bin_widths
    )
