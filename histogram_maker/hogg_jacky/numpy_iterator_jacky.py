"""Python module containing functions that implement a numpy and a iterator-based jackknife-likelihood-computing function to select the optimal bin width among different input (constant) bin widths when computing a histogram.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import modules
import logging
import sys
import numpy as np

# import type aliases from separate sub-module named 'typing_info'
from .typing_info import D_arr, D_list, I_arr, I_list, I_var, D_var, Dtype_var

# import all shared functions and modules (e.g. numpy as np) from separate sub-module named 'shared'
from .shared import *


# set up a logger
logger = logging.getLogger(__name__)


def _get_likelihood_shape(
    my_alpha_parameters: D_arr | D_list, my_bin_numbers: I_arr | I_list
) -> tuple[I_var]:
    """Retrieves the relevant shapes for the array that will hold likelihood data.

    Parameters
    ----------
    my_alpha_parameters : D_arr | D_list
        Contains the alpha smoothing parameters for the Hogg(2008) jackknife likelihood.
    my_bin_numbers : I_arr | I_list
        Contains several numbers of bins to be tried when computing the jackknife likelihood.

    Returns
    -------
    tuple[I_var]
        Contains the relevant shape for the array that will hold likelihood data.
    """
    return (len(my_alpha_parameters) * len(my_bin_numbers),)


def _initialize_empty_array(shape: tuple[I_var], dtype: Dtype_var) -> D_arr:
    """Initializes an empty numpy array based on input.

    Parameters
    ----------
    shape : tuple[I_var]
        The shape for the numpy (nd)array.
    dtype : np.ndtype
        The numpy datatype for the numpy (nd)array.

    Returns
    -------
    D_arr
        Empty numpy (nd)array.
    """
    return np.empty(shape=shape, dtype=dtype)


def _compute_bin_values(my_data: D_arr, bins: I_var) -> D_arr:
    """Computes the bin values for the computation of the jackknife likelihood.

    Parameters
    ----------
    my_data : D_arr
        Data array for which jackknife likelihood is to be computed.
    bins : I_var
        Nr. of bins.

    Returns
    -------
    D_arr
        Bin values for a specific configuration.
    """
    return np.histogram(my_data, bins=bins)[0]


def _compute_empirical_probability_conversion_factor(
    nr_data_points: I_var, nr_bins: I_var, alpha_val: D_var
) -> D_var:
    """Computes the empirical probability conversion factor based on the bin values.

    Parameters
    ----------
    nr_data_points : I_var
        The number of data points to be binned.
    nr_bins : I_var
        The number of bins used.
    alpha_val : D_var
        Alpha smoothing parameter for a specific configuration.

    Returns
    -------
    D_var
        Empirical probability conversion factor.
    """
    return float(nr_data_points) - 1.0 + (float(nr_bins) * alpha_val)


def _compute_jacky(
    bin_values: D_arr,
    bin_width: D_var,
    alpha_val: D_var,
    empirical_probability_conversion_factor: D_var,
) -> D_var:
    """Computes the jackknife likelihood for a specific configuration according to Hogg (2008).

    Parameters
    ----------
    bin_values : D_arr
        Bin values for a specific configuration.
    bin_width : D_var
        Bin width for a specific configuration.
    alpha_val : D_var
        Alpha smoothing parameter for a specific configuration.
    empirical_probability_conversion_factor : D_var
        Empirical probability conversion factor for a specific configuration.

    Returns
    -------
    D_var
        Jackknife likelihood for a specific configuration.
    """
    # compute the value of which a log must be taken
    my_log_input = (bin_values + alpha_val - 1.0) / (
        empirical_probability_conversion_factor * bin_width
    )
    # compute the jackknife likelihood and return it
    return np.vdot(bin_values, np.log(my_log_input))


def _compute_jacky_log_scale(
    bin_values: D_arr,
    bin_edges: D_arr,
    alpha_val: D_var,
    empirical_probability_conversion_factor: D_var,
) -> D_var:
    """Computes the jackknife likelihood for a specific configuration according to Hogg (2008), for log-valued histograms.

    Parameters
    ----------
    bin_values : D_arr
        Bin values for a specific configuration.
    bin_edges : D_arr
        Bin edges for a specific configuration.
    alpha_val : D_var
        Alpha smoothing parameter for a specific configuration.
    empirical_probability_conversion_factor : D_var
        Empirical probability conversion factor for a specific configuration.

    Returns
    -------
    D_var
        Jackknife likelihood for a specific configuration.
    """
    # compute bin width factor
    bw_arr_factor = (
        bin_edges[1:] - bin_edges[:-1]
    ) * empirical_probability_conversion_factor
    # compute value of which a log must be taken
    log_value = np.divide(bin_values + (alpha_val - 1.0), bw_arr_factor)
    # compute the jackknife likelihood and return it
    return np.vdot(bin_values, np.log(log_value))


def _get_empirical_probability_histogram(
    count_histogram: I_arr, data_shape: I_var
) -> D_arr:
    """Computes the empirical probability histogram based on the count histogram.

    Parameters
    ----------
    count_histogram : I_arr
        Count histogram.
    data_shape : I_var
        Shape of the valued data array.

    Returns
    -------
    D_arr
        Empirical probability histogram.
    """
    return count_histogram.astype(np.float64, copy=False) / data_shape


def _get_empirical_density_histogram(
    count_histogram: I_arr, data_shape: I_var, my_bw: D_var or D_arr
) -> D_arr:
    """Computes the empirical density histogram based on the count histogram.

    Parameters
    ----------
    count_histogram : I_arr
        Count histogram.
    data_shape : I_var
        Shape of the valued data array.
    my_bw: D_var or D_arr
        The bin width(s).

    Returns
    -------
    D_arr
        Empirical density histogram.
    """
    return count_histogram.astype(np.float64, copy=False) / (data_shape * my_bw)


def hogg_jackknife_likelihood_numpy(
    my_data: D_arr,
    my_alpha_parameters: D_arr | D_list,
    my_bin_numbers: I_arr | I_list,
    custom_bin_width: D_list = [1.0],
    verbose: bool = False,
    empirical_probability: bool = True,
    empirical_density: bool = False,
    return_binning_parameters: bool = False,
) -> tuple[D_arr | None, D_arr | None, D_var | None]:
    """Defines how the jackknife likelihood is computed using numpy for the Hogg (2008) bin width selection.

    Notes
    -----
    Will select a constant bin width!

    Parameters
    ----------
    my_data: D_arr
        Contains the data that needs to be binned.
    my_alpha_parameters: None | D_arr | D_list
        Contains the alpha smoothing parameters for the Hogg(2008) jackknife likelihood.
    my_bin_numbers: None | I_arr | I_list
        Contains several numbers of bins to be tried when computing the jackknife likelihood.
    custom_bin_width: D_list, optional
        The custom bin width that will be used to generate the bin width if only a single point is present in the histogram (cannot compute likelihood...); by default [1.].
    verbose : bool, optional
        If True, print verbose output; by default False.
    empirical_probability : bool, optional
        If True, retrieve an empirical probability histogram. If False, and 'empirical_density' is True, retrieve an empirical density histogram. If False, and 'empirical_density' is False, retrieve a count histogram; by default True.
    empirical_density : bool, optional
        If True, and 'empirical_probability' is False, retrieve an empirical density histogram. If True or False, and 'empirical_probability' is True, retrieve an empirical probability histogram. If False, and 'empirical_probability' is False, retrieve a count histogram; by default False.
    return_binning_parameters : bool, optional
        If True, retrieve the optimal binning phase, binning alpha (smoothing parameter), and the nr. of bins. If False, do not retrieve such parameters; by default False.

    Returns
    -------
    hist_counts : D_arr
        The count or empirical probability or empirical density histogram that have maximized the jackknife likelihood.
    hist_edges : D_arr
        The bin edges of the histogram whose counts have maximized the jackknife likelihood.
    max_bin_width : D_var | None
        The bin width for the jackknife likelihood is maximized.
    binning_params : dict | None
        Contains the binning parameters, if requested ('return_binning_parameters' == True) and if the optimization was performed, otherwise, None.
    """
    # verify you have data, alpha parameters, and bin numbers available
    if not check_input_data(
        my_data=my_data,
        my_bin_numbers=my_bin_numbers,
        my_alpha_parameters=my_alpha_parameters,
    ):
        # NOT ENOUGH INFO AVAILABLE: EXIT
        sys.exit()
    else:
        # obtain the range of the parameter to be binned, if possible
        if not check_data_shape(my_data=my_data):
            return None, None, None, None
        elif (
            my_single_data_point_out_vals := handle_single_data_point(
                my_data=my_data, custom_bin_width=custom_bin_width
            )
        )[0] is not None:
            my_custom_bin_width, _ones_combo = my_single_data_point_out_vals
            if verbose:
                print_single_value(
                    custom_bin_width=my_custom_bin_width, ones_combo=_ones_combo
                )
            return *np.histogram(my_data, bins=1), my_custom_bin_width, None
        else:
            # get NaN mask
            _nan_mask = np.isnan(my_data)
            # check if any non-NaN values are present, and compute histogram only if there are non-NaN values
            if np.product(_nan_mask):
                # ONLY NaN values: return Nones
                return None, None, None, None
            else:
                # get non-NaN data
                _valued_data = retrieve_valued_data(
                    my_data=my_data, nan_mask=_nan_mask
                )
                # get data range
                _my_data_range = get_peak_to_peak_range(my_data=_valued_data)
                # get the likelihood shape
                _likelihood_shape = _get_likelihood_shape(
                    my_bin_numbers=my_bin_numbers,
                    my_alpha_parameters=my_alpha_parameters,
                )
                # initialize empty array
                _my_likelihoods: D_arr = _initialize_empty_array(
                    shape=_likelihood_shape, dtype=np.float64
                )
                _my_bin_widths: D_arr = _initialize_empty_array(
                    shape=_likelihood_shape, dtype=np.float64
                )
                # generate the list of the combinations over all the available parameters
                _my_combinations = get_parameter_combinations(
                    my_alpha_parameters=my_alpha_parameters,
                    my_bin_numbers=my_bin_numbers,
                )
                # compute the jackknife likelihoods
                for _nr, (_my_alpha, _my_bin_number) in enumerate(
                    _my_combinations
                ):
                    # compute the bin width and store it
                    _bin_width = compute_bin_width(
                        my_ptp_range=_my_data_range,
                        my_bin_number=_my_bin_number,
                    )
                    _my_bin_widths[_nr] = _bin_width
                    # get the bin values for this configuration
                    _my_bin_values = _compute_bin_values(
                        my_data=_valued_data, bins=_my_bin_number
                    )
                    # compute the empirical probability conversion factor
                    _empirical_probability_conversion = (
                        _compute_empirical_probability_conversion_factor(
                            nr_data_points=_valued_data.shape[0],
                            nr_bins=_my_bin_number,
                            alpha_val=_my_alpha,
                        )
                    )
                    # compute the jackknife likelihood and store it
                    _my_jackknife_likelihood = _compute_jacky(
                        bin_values=_my_bin_values,
                        bin_width=_bin_width,
                        alpha_val=_my_alpha,
                        empirical_probability_conversion_factor=_empirical_probability_conversion,
                    )
                    _my_likelihoods[_nr] = _my_jackknife_likelihood
                # define the index pointing to the max likelihood
                _max_idx: I_var = np.nanargmax(_my_likelihoods, axis=0)
                # obtain the bin width and combination of parameters that have the maximized likelihood
                max_bin_width = _my_bin_widths[_max_idx]
                _max_combo = _my_combinations[_max_idx]  # type: ignore
                # if verbose: print output
                if verbose:
                    print_maximal_values(
                        max_bin_width=max_bin_width, max_combo=_max_combo
                    )
                # compute the optimal histogram and corresponding bin edges
                hist_counts, hist_edges = np.histogram(
                    _valued_data, bins=_max_combo[1]
                )
                # store the optimal binning parameters, if needed
                if return_binning_parameters:
                    binning_params = handle_return_binning_parameters(
                        combination_obj=_max_combo
                    )
                else:
                    binning_params = None
                # perform conversion to empirical probabilities or densities if necessary and return the optimal histogram and corresponding bin edges and bin width
                if empirical_probability:
                    return (
                        _get_empirical_probability_histogram(
                            count_histogram=hist_counts,
                            data_shape=_valued_data.shape[0],
                        ),
                        hist_edges,
                        max_bin_width,
                        binning_params,
                    )
                elif empirical_density:
                    return (
                        _get_empirical_density_histogram(
                            count_histogram=hist_counts,
                            data_shape=_valued_data.shape[0],
                            my_bw=max_bin_width,
                        ),
                        hist_edges,
                        max_bin_width,
                        binning_params,
                    )
                else:
                    return (
                        hist_counts,
                        hist_edges,
                        max_bin_width,
                        binning_params,
                    )


def hogg_jackknife_likelihood_iterator(
    my_data: D_arr,
    my_alpha_parameters: D_arr | D_list,
    my_bin_numbers: I_arr | I_list,
    custom_bin_width: D_list = [1.0],
    verbose: bool = False,
    empirical_probability: bool = True,
    empirical_density: bool = False,
    return_binning_parameters: bool = False,
) -> tuple[D_arr | None, D_arr | None, D_var | None]:
    """Defines how the jackknife likelihood is computed for the Hogg (2008) bin width selection that makes use of a python iterator and does not make use of intermediate numpy arrays.

    Notes
    -----
    Will select a constant bin width!

    Parameters
    ----------
    my_data: D_arr
        Contains the data that needs to be binned.
    my_alpha_parameters: D_arr | D_list
        Contains the alpha smoothing parameters for the Hogg(2008) jackknife likelihood.
    my_bin_numbers: I_arr | I_list
        Contains several numbers of bins to be tried when computing the jackknife likelihood.
    custom_bin_width: D_list, optional
        The custom bin width that will be used to generate the bin width if only a single point is present in the histogram (cannot compute likelihood...); by default [1.].
    verbose : bool, optional
        If True, print verbose output; by default False.
    empirical_probability : bool, optional
        If True, retrieve an empirical probability histogram. If False, and 'empirical_density' is True, retrieve an empirical density histogram. If False, and 'empirical_density' is False, retrieve a count histogram; by default True.
    empirical_density : bool, optional
        If True, and 'empirical_probability' is False, retrieve an empirical density histogram. If True or False, and 'empirical_probability' is True, retrieve an empirical probability histogram. If False, and 'empirical_probability' is False, retrieve a count histogram; by default False.
    return_binning_parameters : bool, optional
        If True, retrieve the optimal binning phase, binning alpha (smoothing parameter), and the nr. of bins. If False, do not retrieve such parameters; by default False.

    Returns
    -------
    hist_counts : D_arr | None
        The count or empirical probability or empirical density histogram that have maximized the jackknife likelihood.
    hist_edges : D_arr | None
        The bin edges of the histogram whose counts have maximized the jackknife likelihood.
    max_bin_width : D_var | None
        The bin width for the jackknife likelihood is maximized.
    binning_params : dict | None
        Contains the binning parameters, if requested ('return_binning_parameters' == True) and if the optimization was performed, otherwise, None.
    """
    # verify you have data, alpha parameters, and bin numbers available
    if not check_input_data(
        my_data=my_data,
        my_bin_numbers=my_bin_numbers,
        my_alpha_parameters=my_alpha_parameters,
    ):
        # NOT ENOUGH INFO AVAILABLE: EXIT
        sys.exit()
    else:
        # obtain the range of the parameter to be binned, if possible
        if not check_data_shape(my_data=my_data):
            return None, None, None, None
        elif (
            my_single_data_point_out_vals := handle_single_data_point(
                my_data=my_data, custom_bin_width=custom_bin_width
            )
        )[0] is not None:
            my_custom_bin_width, _ones_combo = my_single_data_point_out_vals
            if verbose:
                print_single_value(
                    custom_bin_width=my_custom_bin_width, ones_combo=_ones_combo
                )
            return *np.histogram(my_data, bins=1), my_custom_bin_width, None
        else:
            # get NaN mask
            _nan_mask = np.isnan(my_data)
            # check if any non-NaN values are present, and compute histogram only if there are non-NaN values
            if np.product(_nan_mask):
                # ONLY NaN values: return Nones
                return None, None, None, None
            else:
                # get non-NaN data
                _valued_data = retrieve_valued_data(
                    my_data=my_data, nan_mask=_nan_mask
                )
                # get data range
                _my_data_range = get_peak_to_peak_range(my_data=_valued_data)
                # initialize return parameters that will continuously be updated
                my_bin_width = custom_bin_width[0]
                my_combination = (1.0, 1)
                # initialize a likelihood parameter used during iteration
                _iter_likelihood = -np.inf
                # generate the iterator over the combinations over all the available parameters
                _my_combinations_iter = get_parameter_combinations(
                    my_alpha_parameters=my_alpha_parameters,
                    my_bin_numbers=my_bin_numbers,
                    return_list=False,
                )
                # compute the jackknife likelihoods, and update the output values based on the likelihoods
                for _my_alpha, _my_bin_number in _my_combinations_iter:
                    # compute the bin width
                    _bin_width = compute_bin_width(
                        my_ptp_range=_my_data_range,
                        my_bin_number=_my_bin_number,
                    )
                    # get the bin values for this configuration
                    _my_bin_values = _compute_bin_values(
                        my_data=_valued_data, bins=_my_bin_number
                    )
                    # compute the empirical probability conversion factor
                    _empirical_probability_conversion = (
                        _compute_empirical_probability_conversion_factor(
                            nr_data_points=_valued_data.shape[0],
                            nr_bins=_my_bin_number,
                            alpha_val=_my_alpha,
                        )
                    )
                    # compute the jackknife likelihood
                    _my_jackknife_likelihood = _compute_jacky(
                        bin_values=_my_bin_values,
                        bin_width=_bin_width,
                        alpha_val=_my_alpha,
                        empirical_probability_conversion_factor=_empirical_probability_conversion,
                    )
                    # verify if the likelihood is larger than the current likelihood
                    if _my_jackknife_likelihood > _iter_likelihood:
                        # replace the iter_likelihood value
                        _iter_likelihood = _my_jackknife_likelihood
                        # LARGER: update values
                        my_bin_width = _bin_width
                        my_combination = (_my_alpha, _my_bin_number)
                # obtain the bin width and combination of parameters that have the maximized likelihood and print them, if verbose == True
                if verbose:
                    print_maximal_values(
                        max_bin_width=my_bin_width, max_combo=my_combination
                    )
                # compute the optimal histogram and corresponding bin edges
                hist_counts, hist_edges = np.histogram(
                    _valued_data, bins=my_combination[1]
                )
                # store the optimal binning parameters, if needed
                if return_binning_parameters:
                    binning_params = handle_return_binning_parameters(
                        combination_obj=my_combination
                    )
                else:
                    binning_params = None
                # perform conversion to empirical probabilities or densities if necessary and return the optimal histogram and corresponding bin edges and bin width
                if empirical_probability:
                    return (
                        _get_empirical_probability_histogram(
                            count_histogram=hist_counts,
                            data_shape=_valued_data.shape[0],
                        ),
                        hist_edges,
                        my_bin_width,
                        binning_params,
                    )
                elif empirical_density:
                    return (
                        _get_empirical_density_histogram(
                            count_histogram=hist_counts,
                            data_shape=_valued_data.shape[0],
                            my_bw=my_bin_width,
                        ),
                        hist_edges,
                        my_bin_width,
                        binning_params,
                    )
                else:
                    return hist_counts, hist_edges, my_bin_width, binning_params


def hogg_jackknife_likelihood_iterator_log(
    my_data: D_arr,
    my_alpha_parameters: D_arr | D_list,
    my_bin_numbers: I_arr | I_list,
    custom_bin_width: D_list = [1.0],
    verbose: bool = False,
    empirical_probability: bool = True,
    empirical_density: bool = False,
    negative_values: bool = False,
    return_binning_parameters: bool = False,
) -> tuple[D_arr | None, D_arr | None, D_arr | None]:
    """Defines how the jackknife likelihood is computed for the Hogg (2008) bin width selection that makes use of a python iterator for a constant width-in-log-space histogram.

    Notes
    -----
    Will select a constant bin width!

    Parameters
    ----------
    my_data: D_arr
        Contains the data that needs to be binned.
    my_alpha_parameters: D_arr | D_list
        Contains the alpha smoothing parameters for the Hogg(2008) jackknife likelihood.
    my_bin_numbers: I_arr | I_list
        Contains several numbers of bins to be tried when computing the jackknife likelihood.
    custom_bin_width: D_list, optional
        The custom bin width that will be used to generate the bin width if only a single point is present in the histogram (cannot compute likelihood...); by default [1.].
    verbose : bool, optional
        If True, print verbose output; by default False.
    empirical_probability : bool, optional
        If True, retrieve an empirical probability histogram. If False, and 'empirical_density' is True, retrieve an empirical density histogram. If False, and 'empirical_density' is False, retrieve a count histogram; by default True.
    empirical_density : bool, optional
        If True, and 'empirical_probability' is False, retrieve an empirical density histogram. If True or False, and 'empirical_probability' is True, retrieve an empirical probability histogram. If False, and 'empirical_probability' is False, retrieve a count histogram; by default False.
    negative_values : bool, optional
        If True, return Nones. If False, assume all values are positive and no shifting is done; by default False.
    return_binning_parameters : bool, optional
        If True, retrieve the optimal binning phase, binning alpha (smoothing parameter), and the nr. of bins. If False, do not retrieve such parameters; by default False.

    Returns
    -------
    hist_counts : D_arr | None
        The count or empirical probability or empirical density histogram that have maximized the jackknife likelihood.
    hist_edges : D_arr | None
        The bin edges of the histogram whose counts have maximized the jackknife likelihood.
    max_bin_widths : D_arr | None
        The bin widths for the jackknife likelihood is maximized.
    binning_params : dict | None
        Contains the binning parameters, if requested ('return_binning_parameters' == True) and if the optimization was performed, otherwise, None.
    """
    # verify you have data, alpha parameters, and bin numbers available
    if not check_input_data(
        my_data=my_data,
        my_bin_numbers=my_bin_numbers,
        my_alpha_parameters=my_alpha_parameters,
    ):
        # NOT ENOUGH INFO AVAILABLE: EXIT
        sys.exit()
    else:
        # obtain the range of the parameter to be binned, if possible
        if not check_data_shape(my_data=my_data):
            return None, None, None, None
        elif (
            my_single_data_point_out_vals := handle_single_data_point(
                my_data=my_data, custom_bin_width=custom_bin_width
            )
        )[0] is not None:
            my_custom_bin_width, _ones_combo = my_single_data_point_out_vals
            if verbose:
                print_single_value(
                    custom_bin_width=my_custom_bin_width, ones_combo=_ones_combo
                )
            return *np.histogram(my_data, bins=1), my_custom_bin_width, None
        else:
            # get NaN mask
            _nan_mask = np.isnan(my_data)
            # check if any non-NaN values are present, and compute histogram only if there are non-NaN values
            if np.product(_nan_mask):
                # ONLY NaN values: return Nones
                return None, None, None, None
            elif negative_values:
                return None, None, None, None
            else:
                # get non-NaN data
                _valued_data = retrieve_valued_data(
                    my_data=my_data, nan_mask=_nan_mask
                )
                _valued_log_data = np.log10(_valued_data)
                # initialize return parameters that will continuously be updated
                my_bin_edges = np.array([])
                my_combination = (np.nan, 0)
                # initialize a likelihood parameter used during iteration
                _iter_likelihood = -np.inf
                # generate the iterator over the combinations over all the available parameters
                _my_combinations_iter = get_parameter_combinations(
                    my_alpha_parameters=my_alpha_parameters,
                    my_bin_numbers=my_bin_numbers,
                    return_list=False,
                )
                # compute the jackknife likelihoods, and update the output values based on the likelihoods
                for _my_alpha, _my_bin_number in _my_combinations_iter:
                    # get the bin values for this configuration
                    (
                        _my_bin_values,
                        _my_bin_edges,
                    ) = histogram_log_counts_and_edges(
                        my_log_data=_valued_log_data,
                        my_data=_valued_data,
                        bin_nr=_my_bin_number,
                        bin_phase=0.0,
                    )
                    # compute the empirical probability conversion factor
                    _empirical_probability_conversion = (
                        _compute_empirical_probability_conversion_factor(
                            nr_data_points=_valued_data.shape[0],
                            nr_bins=_my_bin_number,
                            alpha_val=_my_alpha,
                        )
                    )
                    # compute the jackknife likelihood
                    _my_jackknife_likelihood = _compute_jacky_log_scale(
                        bin_values=_my_bin_values,
                        bin_edges=_my_bin_edges,
                        alpha_val=_my_alpha,
                        empirical_probability_conversion_factor=_empirical_probability_conversion,
                    )
                    # verify if the likelihood is larger than the current likelihood
                    if _my_jackknife_likelihood > _iter_likelihood:
                        # replace the iter_likelihood value
                        _iter_likelihood = _my_jackknife_likelihood
                        # LARGER: update values
                        my_bin_edges = _my_bin_edges
                        my_combination = (_my_alpha, _my_bin_number)
                # obtain the bin width and combination of parameters that have the maximized likelihood and print them, if verbose == True
                if verbose:
                    print_maximal_values(
                        max_bin_width_edges=my_bin_edges,
                        max_combo=my_combination,
                    )
                # compute the optimal histogram and corresponding bin edges
                hist_counts, hist_edges = histogram_log_counts_and_edges(
                    my_log_data=np.log10(_valued_data),
                    my_data=_valued_data,
                    bin_nr=my_combination[1],
                    bin_phase=0.0,
                )
                # store the optimal binning parameters, if needed
                if return_binning_parameters:
                    binning_params = handle_return_binning_parameters(
                        combination_obj=my_combination
                    )
                else:
                    binning_params = None
                # perform conversion to empirical probabilities or densities if necessary and return the optimal histogram and corresponding bin edges and bin width
                if empirical_probability:
                    return (
                        _get_empirical_probability_histogram(
                            count_histogram=hist_counts,
                            data_shape=_valued_data.shape[0],
                        ),
                        hist_edges,
                        hist_edges[1:] - hist_edges[:-1],
                        binning_params,
                    )
                elif empirical_density:
                    bw_opt = hist_edges[1:] - hist_edges[:-1]
                    return (
                        _get_empirical_density_histogram(
                            count_histogram=hist_counts,
                            data_shape=_valued_data.shape[0],
                            my_bw=bw_opt,
                        ),
                        hist_edges,
                        bw_opt,
                        binning_params,
                    )
                else:
                    return (
                        hist_counts,
                        hist_edges,
                        hist_edges[1:] - hist_edges[:-1],
                        binning_params,
                    )


def histogram_spec_width(
    my_data: D_arr | None,
    bin_nr: I_var | None,
    empirical_probability: bool,
    empirical_density: bool = False,
) -> D_arr:
    """Individual function to be used to compute a constant-width histogram of a data set with a specified bin width.

    Parameters
    ----------
    my_data : D_arr | None
        Data array for which constant-width histogram is to be computed.
    bin_nr : I_var
        Nr. of bins. None if the bin width is None.
    empirical_probability : bool
        If True, retrieve an empirical probability histogram. If False, and 'empirical_density' is True, retrieve an empirical density histogram. If False, and 'empirical_density' is False, retrieve a count histogram.
    empirical_probability : bool, optional
        If True, and 'empirical_probability' is False, retrieve an empirical density histogram. If True or False, and 'empirical_density' is True, retrieve an empirical density histogram. If False, and 'empirical_density' is False, retrieve a count histogram; by default False.

    Returns
    -------
    D_arr | None
        Bin counts for the constant-width histogram. None if the input array 'my_data' does not have any entries or if the bin number is None (i.e. the specified band width is None).
    """
    # GUARD FOR BIN WIDTH SPECIFIED AS NONE: NO BIN NR AVAILABLE, OR IF MY_DATA IS SPECIFIED AS NONE
    if bin_nr is None or my_data is None:
        return None
    # GUARD FOR ARRAY WITH ZERO ENTRIES
    elif my_data.shape[0] == 0:
        return None
    else:
        # compute the optimal histogram
        hist_counts, bin_edges = np.histogram(my_data, bins=bin_nr)
        # return the optimal histogram counts or its empirical probability or empirical density equivalent
        if empirical_probability:
            return _get_empirical_probability_histogram(
                count_histogram=hist_counts, data_shape=my_data.shape[0]
            )
        elif empirical_density:
            return _get_empirical_density_histogram(
                count_histogram=hist_counts,
                data_shape=my_data.shape[0],
                my_bw=bin_edges[1] - bin_edges[0],
            )
        else:
            return hist_counts


def histogram_log_spec_edges(
    my_data: D_arr | None,
    my_bin_edges: D_arr | None,
    empirical_probability: bool,
    empirical_density: bool = False,
    min_nr_bins: I_var = 0,
    negative_values: bool = False,
) -> D_arr | None:
    """Individual function to be used to compute a histogram with bins of constant width in log space, as determined from specified bin edges.

    Parameters
    ----------
    my_data : D_arr | None
        Data array for which a histogram is to be computed with constant-width bins in log space.
    my_bin_edges : D_arr | None
        Specified bin edges. None values will cause this function to return None as the bin counts.
    empirical_probability : bool
        If True, retrieve an empirical probability histogram. If False, and 'empirical_density' is True, retrieve an empirical density histogram. If False, and 'empirical_density' is False, retrieve a count histogram.
    empirical_probability : bool, optional
        If True, and 'empirical_probability' is False, retrieve an empirical density histogram. If True or False, and 'empirical_density' is True, retrieve an empirical density histogram. If False, and 'empirical_density' is False, retrieve a count histogram; by default False.
    min_nr_bins : I_var, optional
        The minimum number of bins to be considered; by default 0.
    negative_values : bool, optional
        If True, return None. If False, assume all values are positive and no shifting is done; by default False.

    Returns
    -------
    D_arr | None
        Bin counts for the histogram with constant-width bins in log space. None if the input array 'my_data' does not have any entries or if the specified 'my_bin_edges' or 'my_data' are None.
    """
    # GUARD FOR BIN_EDGES OR DATA SPECIFIED AS NONE
    if my_bin_edges is None or my_data is None:
        return None
    # GUARD FOR ARRAY WITH ZERO ENTRIES
    elif my_data.shape[0] == 0:
        return None
    elif negative_values:
        return None
    else:
        if empirical_probability:
            return empirical_probability_histogram_log_counts(
                my_data=my_data,
                my_bin_edges=my_bin_edges,
                binning_phase=0.0,
                min_nr_bins=min_nr_bins,
            )
        elif empirical_density:
            return empirical_density_histogram_log_counts(
                my_data=my_data,
                my_bin_edges=my_bin_edges,
                binning_phase=0.0,
                min_nr_bins=min_nr_bins,
            )
        else:
            return histogram_log_counts(
                my_data=my_data,
                my_bin_edges=my_bin_edges,
                binning_phase=0.0,
                min_nr_bins=min_nr_bins,
            )
