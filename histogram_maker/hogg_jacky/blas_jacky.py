"""Python module containing functions that implement a BLAS-empowered jackknife-likelihood-computing function to to select the optimal bin width among different input (constant) bin widths when computing a histogram.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import modules
import logging
import sys
import numpy as np
from scipy.linalg.blas import daxpy, dscal, dasum, ddot  # type: ignore

# import type aliases from separate sub-module named 'typing_info'
from .typing_info import D_arr, D_var, I_var, I_arr, D_list, I_list

# import all shared functions and modules (e.g. numpy as np) from separate sub-module named 'shared'
from .shared import *


# set up a logger
logger = logging.getLogger(__name__)


def histogram_blas(
    my_data: D_arr,
    bin_width: D_var,
    bin_nr: I_var,
    bin_phase: D_var = 0.0,
    log_bins: bool = False,
    min_nr_bins: I_var = 0,
    negative_values: bool = False,
    additional_checks: bool = False,
    my_data_min: D_var | None = None,
) -> tuple[D_arr | None, D_arr | None, D_var | D_arr | None]:
    """Computes a constant-bin-width histogram using BLAS functionalities wrapped in Scipy.

    Parameters
    ----------
    my_data : D_arr
        Contains the data (or log10(data) if 'log_bins' is True) that needs to be binned.
    bin_width : D_var
        Constant (non-variable) bin width.
    bin_nr : I_var
        Nr. of bins to be computed.
    bin_phase : D_var, optional
        Bin phase factor (see Hogg, 2008); by default 0.0.
    log_bins : bool, optional
        If True, perform binning in a log10 space (convert 'my_data' to log10('my_data') before computing the histogram!). If False, perform binning in a linear space; by default False.
    min_nr_bins : I_var, optional
        The minimum number of bins to be considered; by default 0.
    negative_values : bool, optional
        If True, assert that all actions can be taken with negative values in the data array 'my_data' (log transformation not possible!). If False, assume all values in the data array 'my_data' are positive; by default False.
    additional_checks : bool, optional
        Performs additional checks if requested (True); by default False.
    my_data_min : D_var | None, optional
        Specific minimum data value for the binning if not None. If None, use the minimum value of 'my_data'; by default None. Only of importance when not sampling logarithmic data!

    Returns
    -------
    tuple[D_arr | None, D_arr | None, D_var | D_arr | None]
        Histogram counts, corresponding bin edges and corresponding bin width(s).
    """
    if log_bins and negative_values:
        return None, None, None
    elif log_bins:
        hist_counts, hist_edges = histogram_log_counts_and_edges(
            my_log_data=np.log10(my_data),
            my_data=my_data,
            bin_nr=bin_nr,
            bin_phase=bin_phase,
            min_nr_bins=min_nr_bins,
        )
        return (
            hist_counts,
            hist_edges,
            daxpy(
                hist_edges[:-1],
                hist_edges[1:].copy(),
                hist_edges.shape[0] - 1,
                -1.0,
            ),
        )
    else:
        return (
            _blas_histogram_counts(
                my_data=my_data,
                bin_width=bin_width,
                bin_phase=bin_phase,
                nr_bins=bin_nr,
                min_nr_bins=min_nr_bins,
                additional_checks=additional_checks,
                my_data_min=my_data_min,
            ),
            _bin_edges(my_data=my_data, bin_nr=bin_nr),
            bin_width,
        )


def empirical_probability_histogram_blas(
    my_data: D_arr,
    bin_width: D_var,
    bin_nr: I_var,
    bin_phase: D_var = 0.0,
    log_bins: bool = False,
    min_nr_bins: I_var = 0,
    negative_values: bool = False,
    additional_checks: bool = False,
    my_data_min: D_var | None = None,
) -> tuple[D_arr | None, D_arr | None, D_var | D_arr | None]:
    """Computes a empirical probability constant-bin-width histogram using BLAS functionalities wrapped in Scipy.

    Parameters
    ----------
    my_data : D_arr
        Contains the data (or log10(data) if 'log_bins' is True) that needs to be binned.
    bin_width : D_var
        Constant (non-variable) bin width.
    bin_nr : I_var
        Nr. of bins to be computed.
    bin_phase : D_var, optional
        Bin phase factor (see Hogg, 2008); by default 0.0.
    log_bins : bool, optional
        If True, perform binning in a log10 space (convert 'my_data' to log10('my_data') before computing the histogram!). If False, perform binning in a linear space; by default False.
    min_nr_bins : I_var, optional
        The minimum number of bins to be considered; by default 0.
    negative_values : bool, optional
        If True, shift all values in 'my_data' with the lowest-value entry of 'my_data' (i.e. rendering all elements of the new array positive for binning calculations) and handle that shifted value for output. If False, assume all values are positive and no shifting is done; by default False.
    additional_checks : bool, optional
        Performs additional checks if requested (True); by default False.
    my_data_min : D_var | None, optional
        Specific minimum data value for the binning if not None. If None, use the minimum value of 'my_data'; by default None.

    Returns
    -------
    tuple[D_arr | None, D_arr | None, D_var | D_arr | None]
        Empirical probability histogram counts, corresponding bin edges and corresponding bin width(s).
    """
    # get count histogram and edges
    count_histogram, bin_edges, bin_width = histogram_blas(
        my_data=my_data,
        bin_width=bin_width,
        bin_nr=bin_nr,
        bin_phase=bin_phase,
        log_bins=log_bins,
        min_nr_bins=min_nr_bins,
        negative_values=negative_values,
        additional_checks=additional_checks,
        my_data_min=my_data_min,
    )
    # compute empirical probability histogram and return it together with the bin edges
    try:
        return (
            dscal(1.0 / my_data.shape[0], count_histogram),
            bin_edges,
            bin_width,
        )
    except AttributeError:
        # returned value from histogram_blas = None, None, None
        return None, None, None


def empirical_density_histogram_blas(
    my_data: D_arr,
    bin_width: D_var,
    bin_nr: I_var,
    bin_phase: D_var = 0.0,
    log_bins: bool = False,
    min_nr_bins: I_var = 0,
    negative_values: bool = False,
    additional_checks: bool = False,
    my_data_min: D_var | None = None,
) -> tuple[D_arr | None, D_arr | None, D_var | D_arr | None]:
    """Computes a empirical density constant-bin-width histogram using BLAS functionalities wrapped in Scipy.

    Parameters
    ----------
    my_data : D_arr
        Contains the data (or log10(data) if 'log_bins' is True) that needs to be binned.
    bin_width : D_var
        Constant (non-variable) bin width.
    bin_nr : I_var
        Nr. of bins to be computed.
    bin_phase : D_var, optional
        Bin phase factor (see Hogg, 2008); by default 0.0.
    log_bins : bool, optional
        If True, perform binning in a log10 space (convert 'my_data' to log10('my_data') before computing the histogram!). If False, perform binning in a linear space; by default False.
    min_nr_bins : I_var, optional
        The minimum number of bins to be considered; by default 0.
    negative_values : bool, optional
        If True, shift all values in 'my_data' with the lowest-value entry of 'my_data' (i.e. rendering all elements of the new array positive for binning calculations) and handle that shifted value for output. If False, assume all values are positive and no shifting is done; by default False.
    additional_checks : bool, optional
        Performs additional checks if requested (True); by default False.
    my_data_min : D_var | None, optional
        Specific minimum data value for the binning if not None. If None, use the minimum value of 'my_data'; by default None.

    Returns
    -------
    tuple[D_arr | None, D_arr | None, D_var | D_arr | None]
        Empirical density histogram counts, corresponding bin edges and corresponding bin width(s).
    """
    # get count histogram and edges
    count_histogram, bin_edges, bin_width = histogram_blas(
        my_data=my_data,
        bin_width=bin_width,
        bin_nr=bin_nr,
        bin_phase=bin_phase,
        log_bins=log_bins,
        min_nr_bins=min_nr_bins,
        negative_values=negative_values,
        additional_checks=additional_checks,
        my_data_min=my_data_min,
    )
    # compute empirical density histogram and return it together with the bin edges
    try:
        return (
            dscal(1.0 / (my_data.shape[0] * bin_width), count_histogram),
            bin_edges,
            bin_width,
        )
    except AttributeError:
        # returned value from histogram_blas = None, None, None
        return None, None, None


def _blas_histogram_counts(
    my_data: D_arr,
    bin_width: D_var,
    bin_phase: D_var = 0.0,
    nr_bins: I_var = 1,
    min_nr_bins: I_var = 0,
    additional_checks: bool = False,
    my_data_min: D_var | None = None,
) -> D_arr:
    """Computes the bare-bone counts of the bins in a constant-bin-width histogram (whose bins have widths specified by 'bin_width'), without computation of bin edges.

    Parameters
    ----------
    my_data : D_arr
        Data array for which jackknife likelihood is to be computed or None.
    bin_width : D_var
        Bin width.
    bin_phase : D_var, optional
        Bin phase factor (see Hogg, 2008); by default 0.0.
    nr_bins : I_var, optional
        The number of bins; by default 1.
    min_nr_bins : I_var, optional
        The minimum number of bins to be considered; by default 0.
    additional_checks : bool, optional
        Performs additional checks if requested (True); by default False.
    my_data_min : D_var | None, optional
        Specific minimum data value for the binning if not None. If None, use the minimum value of 'my_data'; by default None.

    Returns
    -------
    D_arr
        Histogram counts.
    """
    if nr_bins == 1:
        return np.array([my_data.shape[0]], dtype=my_data.dtype)
    else:
        # obtain the binning result
        _binning_result = _blas_binning(
            my_data=my_data,
            bin_width=bin_width,
            bin_phase=bin_phase,
            my_data_min=my_data_min,
        )
        # adjust upper edge cases
        _binning_result[_binning_result == nr_bins] = nr_bins - 1
        # ensure that no negative counts are ever registered (due to mess-up at argmax, for example)
        if additional_checks and (neg_mask := _binning_result < 0).any():
            _idx = np.argwhere(neg_mask)
            if len(_idx) == 1 and _idx[0] == np.argmin(my_data):
                # mistake was made when counting at argmax, reverse it, by setting the counts in this bin to zero!
                _binning_result[neg_mask] = 0
            else:
                logger.error(
                    f'There are negative values at indices {_idx} in my bin array. This is probably because the search grid logspace values do not include the lower boundary of the values in the passed data array (aka: out of binning bounds error). Will now exit!'
                )
                sys.exit()
        # return the bin counts
        return np.bincount(_binning_result, minlength=min_nr_bins).astype(
            np.float64, copy=False
        )


def _blas_binning(
    my_data: D_arr,
    bin_width: D_var,
    bin_phase: D_var = 0.0,
    my_data_min: D_var | None = None,
) -> I_arr:
    """BLAS-empowered binning function for constant-bin-width histograms.

    Notes
    -----
    Warning: because of the implementation, this binning procedure creates a numpy array containing the bin numbers ranging from 0 to 'my_data.shape[0]'. Hence, post-processing of the upper bound values is necessary if a certain number of bins is required for our output (see e.g. the function '_blas_histogram_counts', which is the preferred method of binning using BLAS functionalities defined in SciPy anyway).

    Parameters
    ----------
    my_data : D_arr
        Data array for which jackknife likelihood is to be computed or None.
    bin_width : D_var
        Bin width.
    bin_phase : D_var, optional
        Bin phase factor (see Hogg, 2008); by default 0.0.
    my_data_min : D_var | None, optional
        Specific minimum data value for the binning if not None. If None, use the minimum value of 'my_data'; by default None.

    Returns
    -------
    I_arr
        Bin numbers for a specified bin width.
    """
    # Verify which minimum data point should be used and prepare the data for binning
    prep_data = (
        my_data - my_data.min()
        if (my_data_min is None)
        else my_data - my_data_min
    )
    # return binning array
    return (
        np.floor(
            (
                daxpy(
                    prep_data,
                    np.full_like(my_data, bin_phase),
                    my_data.shape[0],
                    1.0 / bin_width,
                )
            )
        )
        .round()
        .astype(np.int32, copy=False)
    )


def _bin_edges(my_data: D_arr, bin_nr: I_var) -> D_arr:
    """Custom function used to determine bin edges for constant-bin-width histograms.

    Parameters
    ----------
    my_data : D_arr
        Data array for which jackknife likelihood is to be computed.
    bin_nr : I_var
        The nr. of bins in the histogram.

    Returns
    -------
    D_arr
        Edges of the bins for the histogram.
    """
    return np.linspace(
        my_data.min(),
        my_data.max(),
        num=bin_nr + 1,
        endpoint=True,
        dtype=my_data.dtype,
    )


def _blas_empirical_probability_conversion_factor(
    bin_values: D_arr, alpha_val: D_var
) -> D_var:
    """BLAS-empowered function used to compute the empirical probability conversion factor based on the bin values of a constant-bin-width histogram.

    Parameters
    ----------
    bin_values : D_arr
        Bin values for a specific configuration.
    alpha_val : D_var
        Alpha smoothing parameter for a specific configuration.

    Returns
    -------
    D_var
        Empirical probability conversion factor.
    """
    return dasum(daxpy(bin_values, np.full_like(bin_values, alpha_val))) - 1.0


def _my_jacky(
    bin_values: D_arr,
    bin_width: D_var,
    alpha_val: D_var,
    empirical_probability_conversion_factor: D_var,
) -> D_var:
    """Custom function that computes the jackknife likelihood for a specific configuration according to Hogg (2008), using BLAS functions.

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
    my_log_input = dscal(
        1.0 / (empirical_probability_conversion_factor * bin_width),
        daxpy(bin_values, np.full_like(bin_values, alpha_val - 1.0)),
    )
    # compute the jackknife likelihood and return it
    return ddot(bin_values, np.log(my_log_input))


def _my_jacky_log_scale(
    bin_values: D_arr,
    bin_edges: D_arr,
    alpha_val: D_var,
    empirical_probability_conversion_factor: D_var,
) -> D_var:
    """Custom function that computes the jackknife likelihood for a specific configuration according to Hogg (2008), for log-scale bins, using BLAS functions.

    Parameters
    ----------
    bin_values : D_arr
        Bin values for a specific configuration.
    bin_edges: D_arr
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
    # compute the bin width array
    bw_arr = daxpy(
        bin_edges[:-1], bin_edges[1:].copy(), bin_edges.shape[0] - 1, -1.0
    )
    # compute second part of value of which a log must be taken
    second_part_log = dscal(
        1.0 / empirical_probability_conversion_factor,
        daxpy(bin_values, np.full_like(bin_values, alpha_val - 1.0)),
    )
    # compute the jackknife likelihood and return it
    return ddot(bin_values, np.log(np.divide(second_part_log, bw_arr)))


def hogg_jackknife_likelihood_blas(
    my_data: D_arr,
    my_alpha_parameters: D_arr | D_list,
    my_bin_numbers: I_arr | I_list,
    my_bin_phases: D_arr | D_list = [0.0],
    custom_bin_width: D_list = [1.0],
    verbose: bool = False,
    empirical_probability: bool = True,
    empirical_density: bool = False,
    additional_checks: bool = False,
    return_binning_parameters: bool = False,
) -> tuple[D_arr | None, D_arr | None, D_var | None]:
    """Defines how the jackknife likelihood is computed for the Hogg (2008) (constant) bin width selection procedure, using a BLAS-empowered implementation for a constant-width histogram.

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
    my_bin_phases : D_arr | D_list, optional
        Phase factors for the binning process, as determined in Hogg (2008); by default [0.0].
    custom_bin_width: D_list, optional
        The custom bin width that will be used to generate the bin width if only a single point is present in the histogram (cannot compute likelihood...); by default [1.].
    verbose : bool, optional
        If True, print verbose output; by default False.
    empirical_probability : bool, optional
        If True, retrieve an empirical probability histogram. If False, and 'empirical_density' is True, retrieve an empirical density histogram. If False, and 'empirical_density' is False, retrieve a count histogram; by default True.
    empirical_density : bool, optional
        If True, and 'empirical_probability' is False, retrieve an empirical density histogram. If True or False, and 'empirical_probability' is True, retrieve an empirical probability histogram. If False, and 'empirical_probability' is False, retrieve a count histogram; by default False.
    additional_checks : bool, optional
        Performs additional checks if requested (True); by default False.
    return_binning_parameters : bool, optional
        If True, retrieve the optimal binning phase, binning alpha (smoothing parameter), and the nr. of bins. If False, do not retrieve such parameters; by default False.

    Returns
    -------
    hist_counts : D_arr | None
        The count or empirical probability or empirical density histogram that has maximized the jackknife likelihood.
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
                my_combination = (np.nan, 0)
                my_bin_phase: np.float64 = np.nan
                # initialize a likelihood parameter used during iteration
                _iter_likelihood = -np.inf
                # generate the iterator over the combinations over all the available parameters
                _my_combinations_iter = get_parameter_combinations(
                    my_alpha_parameters=my_alpha_parameters,
                    my_bin_numbers=my_bin_numbers,
                    my_bin_phases=my_bin_phases,
                    return_list=False,
                )
                # compute the jackknife likelihoods, and update the output values based on the likelihoods
                for (
                    _my_alpha,
                    _my_bin_number,
                    _bin_phase,
                ) in _my_combinations_iter:
                    # compute the bin width
                    _bin_width = compute_bin_width(
                        my_ptp_range=_my_data_range,
                        my_bin_number=_my_bin_number,
                    )
                    # get the bin values for this configuration
                    _my_bin_values = _blas_histogram_counts(
                        my_data=_valued_data,
                        bin_width=_bin_width,
                        bin_phase=_bin_phase,
                        nr_bins=_my_bin_number,
                        additional_checks=additional_checks,
                    )
                    # compute the empirical probability conversion factor
                    _empirical_probability_conversion = (
                        _blas_empirical_probability_conversion_factor(
                            bin_values=_my_bin_values, alpha_val=_my_alpha
                        )
                    )
                    # compute the jackknife likelihood
                    _my_jackknife_likelihood = _my_jacky(
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
                        my_bin_phase = _bin_phase
                # obtain the bin width and combination of parameters that have the maximized likelihood and print them, if verbose == True
                if verbose:
                    print_maximal_values(
                        max_bin_width=my_bin_width,
                        max_combo=my_combination,
                        max_bin_phase=my_bin_phase,
                    )
                # store the optimal binning parameters, if needed
                if return_binning_parameters:
                    binning_params = handle_return_binning_parameters(
                        combination_obj=my_combination
                    )
                else:
                    binning_params = None
                # compute the optimal histogram, corresponding bin edges, perform conversion to empirical probabilities or densities if necessary and return the optimal histogram and corresponding bin edges and bin width
                if empirical_probability:
                    return *empirical_probability_histogram_blas(
                        my_data=_valued_data,
                        bin_width=my_bin_width,
                        bin_nr=my_combination[1],
                        bin_phase=my_bin_phase,
                        additional_checks=additional_checks,
                    ), binning_params
                elif empirical_density:
                    return *empirical_density_histogram_blas(
                        my_data=_valued_data,
                        bin_width=my_bin_width,
                        bin_nr=my_combination[1],
                        bin_phase=my_bin_phase,
                        additional_checks=additional_checks,
                    ), binning_params
                else:
                    return *histogram_blas(
                        my_data=_valued_data,
                        bin_width=my_bin_width,
                        bin_nr=my_combination[1],
                        bin_phase=my_bin_phase,
                        additional_checks=additional_checks,
                    ), binning_params


def hogg_jackknife_likelihood_blas_log(
    my_data: D_arr,
    my_alpha_parameters: D_arr | D_list,
    my_bin_numbers: I_arr | I_list,
    my_bin_phases: D_arr | D_list = [0.0],
    custom_bin_width: D_list = [1.0],
    verbose: bool = False,
    empirical_probability: bool = True,
    empirical_density: bool = False,
    negative_values: bool = False,
    return_binning_parameters: bool = False,
) -> tuple[D_arr | None, D_arr | None, D_var | None]:
    """Defines how the jackknife likelihood is computed for the Hogg (2008) (constant) bin width selection procedure, using a BLAS-empowered implementation for a constant-width-in-log-space histogram.

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
    my_bin_phases : D_arr | D_list, optional
        Phase factors for the binning process, as determined in Hogg (2008); by default [0.0].
    custom_bin_width: D_list, optional
        The custom bin width that will be used to generate the bin width if only a single point is present in the histogram (cannot compute likelihood...); by default [1.].
    verbose : bool, optional
        If True, print verbose output; by default False.
    empirical_probability : bool, optional
        If True, retrieve an empirical probability histogram. If False, and 'empirical_density' is True, retrieve an empirical density histogram. If False, and 'empirical_density' is False, retrieve a count histogram; by default True.
    empirical_density : bool, optional
        If True, and 'empirical_probability' is False, retrieve an empirical density histogram. If True or False, and 'empirical_probability' is True, retrieve an empirical probability histogram. If False, and 'empirical_probability' is False, retrieve a count histogram; by default False.
    negative_values : bool, optional
        If True, assert that all actions can be taken with negative values in the data array 'my_data' (log transformation not possible!). If False, assume all values in the data array 'my_data' are positive; by default False.
    return_binning_parameters : bool, optional
        If True, retrieve the optimal binning phase, binning alpha (smoothing parameter), and the nr. of bins. If False, do not retrieve such parameters; by default False.

    Returns
    -------
    hist_counts : D_arr | None
        The count or empirical probability or empirical density histogram that has maximized the jackknife likelihood.
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
            elif negative_values:
                return None, None, None, None
            else:
                # get non-NaN log-scale data
                _valued_data = retrieve_valued_data(
                    my_data=my_data, nan_mask=_nan_mask
                )
                _valued_log_data = np.log10(_valued_data)
                # initialize return parameters that will continuously be updated
                my_bin_edges = np.array([])
                my_combination = (np.nan, 0)
                my_bin_phase: np.float64 = np.nan
                # initialize a likelihood parameter used during iteration
                _iter_likelihood = -np.inf
                # generate the iterator over the combinations over all the available parameters
                _my_combinations_iter = get_parameter_combinations(
                    my_alpha_parameters=my_alpha_parameters,
                    my_bin_numbers=my_bin_numbers,
                    my_bin_phases=my_bin_phases,
                    return_list=False,
                )
                # compute the jackknife likelihoods, and update the output values based on the likelihoods
                for (
                    _my_alpha,
                    _my_bin_number,
                    _bin_phase,
                ) in _my_combinations_iter:
                    # get the bin values for this configuration
                    (
                        _my_bin_values,
                        _my_bin_edges,
                    ) = histogram_log_counts_and_edges(
                        my_log_data=_valued_log_data,
                        my_data=_valued_data,
                        bin_nr=_my_bin_number,
                        bin_phase=_bin_phase,
                    )
                    # compute the empirical probability conversion factor
                    _empirical_probability_conversion = (
                        _blas_empirical_probability_conversion_factor(
                            bin_values=_my_bin_values, alpha_val=_my_alpha
                        )
                    )
                    # compute the jackknife likelihood
                    _my_jackknife_likelihood = _my_jacky_log_scale(
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
                        my_bin_phase = _bin_phase
                # obtain the bin width and combination of parameters that have the maximized likelihood and print them, if verbose == True
                if verbose:
                    print_maximal_values(
                        max_bin_width_edges=my_bin_edges,
                        max_combo=my_combination,
                        max_bin_phase=my_bin_phase,
                    )
                # store the optimal binning parameters, if needed
                if return_binning_parameters:
                    binning_params = handle_return_binning_parameters(
                        combination_obj=my_combination
                    )
                else:
                    binning_params = None
                # compute the optimal histogram, corresponding bin edges, perform conversion to empirical probabilities or densities if necessary and return the optimal histogram and corresponding bin edges and bin width
                if empirical_probability:
                    return *empirical_probability_histogram_blas(
                        my_data=_valued_data,
                        bin_width=np.nan,
                        bin_nr=my_combination[1],
                        bin_phase=my_bin_phase,
                        log_bins=True,
                    ), binning_params
                elif empirical_density:
                    return *empirical_density_histogram_blas(
                        my_data=_valued_data,
                        bin_width=np.nan,
                        bin_nr=my_combination[1],
                        bin_phase=my_bin_phase,
                        log_bins=True,
                    ), binning_params
                else:
                    return *histogram_blas(
                        my_data=_valued_data,
                        bin_width=np.nan,
                        bin_nr=my_combination[1],
                        bin_phase=my_bin_phase,
                        log_bins=True,
                    ), binning_params


def histogram_blas_spec_width(
    my_data: D_arr | None,
    my_bw: D_arr | D_var | None,
    bin_nr: I_var,
    bin_phase: D_var,
    empirical_probability: bool,
    empirical_density: bool = False,
    min_nr_bins: I_var = 0,
    additional_checks: bool = False,
    my_data_min: D_var | None = None,
) -> D_arr | None:
    """Individual function to be used to compute a constant-width histogram of a data set with a specified bin width.

    Parameters
    ----------
    my_data : D_arr | None
        Data array for which constant-width histogram is to be computed.
    my_bw : D_arr | D_var | None
        Specified bin width, or array of bin widths, or None.
    bin_nr : I_var
        Nr. of bins.
    bin_phase : D_var
        Binning phase.
    empirical_probability : bool
        If True, retrieve an empirical probability histogram. If False, and 'empirical_density' is True, retrieve an empirical density histogram. If False, and 'empirical_density' is False, retrieve a count histogram.
    empirical_probability : bool, optional
        If True, and 'empirical_probability' is False, retrieve an empirical density histogram. If True or False, and 'empirical_density' is True, retrieve an empirical density histogram. If False, and 'empirical_density' is False, retrieve a count histogram; by default False.
    min_nr_bins : I_var, optional
        The minimum number of bins to be considered; by default 0.
    additional_checks : bool, optional
        Performs additional checks if requested (True); by default False.
    my_data_min : D_var | None, optional
        Specific minimum data value for the binning if not None. If None, use the minimum value of 'my_data'; by default None.

    Returns
    -------
    D_arr | None
        Bin counts for the constant-width histogram. None if the input array 'my_data' does not have any entries or if the specified 'my_bw' or 'my_data' is None.
    """
    # GUARD FOR BIN WIDTH OR DATA SPECIFIED AS NONE
    if my_bw is None or my_data is None:
        return None
    # GUARD FOR ARRAY WITH ZERO ENTRIES
    elif my_data.shape[0] == 0:
        return None
    else:
        if empirical_probability:
            return empirical_probability_histogram_blas(
                my_data=my_data,
                bin_width=my_bw,
                bin_nr=bin_nr,
                bin_phase=bin_phase,
                min_nr_bins=min_nr_bins,
                additional_checks=additional_checks,
                my_data_min=my_data_min,
            )[0]
        elif empirical_density:
            return empirical_density_histogram_blas(
                my_data=my_data,
                bin_width=my_bw,
                bin_nr=bin_nr,
                bin_phase=bin_phase,
                min_nr_bins=min_nr_bins,
                additional_checks=additional_checks,
                my_data_min=my_data_min,
            )[0]
        else:
            return histogram_blas(
                my_data=my_data,
                bin_width=my_bw,
                bin_nr=bin_nr,
                bin_phase=bin_phase,
                min_nr_bins=min_nr_bins,
                additional_checks=additional_checks,
                my_data_min=my_data_min,
            )[0]


def histogram_blas_log_spec_edges(
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
