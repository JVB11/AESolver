"""Python module containing functions shared among different implementations for the jackknife likelihood computations that facilitate handling specific situations when computing histograms and performing optimizations.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import modules
import logging
import sys

# import type aliases from separate modules
from ..typing_info import D_arr, D_list, D_var, I_var


# set up a logger
logger = logging.getLogger(__name__)


# handle case where only 1 data point is in the data array
def handle_single_data_point(
    my_data: D_arr, custom_bin_width: D_list
) -> tuple[D_var, tuple[D_var, I_var]] | tuple[None, None]:
    """Handles the case of a single data point.

    Parameters
    ----------
    my_data : D_arr
        Data array for which jackknife likelihood is to be computed.
    custom_bin_width : D_list
        Custom bin width that will be used if only a single data point is available (because we cannot compute the likelihood in that case!)

    Returns
    -------
    tuple[D_var, tuple[D_var, I_var]] | tuple[None, None]
        Returns parameters for the single data point case or None.
    """
    if my_data.shape[0] == 1:
        return custom_bin_width[0], (1.0, 1)
    else:
        return (None, None)


# handle return of data binning parameters
def handle_return_binning_parameters(combination_obj: list | tuple) -> dict:
    """Reformats the binning parameters into a dictionary from a passed object (list or tuple) that contains those optimal parameters.

    Parameters
    ----------
    combination_obj : list | tuple
        Contains the optimal binning parameters.

    Returns
    -------
    dict
        Stores the optimal binning parameters for additional string formatting later on.
    """
    if (combo_len := len(combination_obj)) == 2:
        return {
            'binning alpha': combination_obj[0],
            'nr of bins': combination_obj[1],
            'binning phase': 0,
        }
    elif combo_len == 3:
        return {
            'binning alpha': combination_obj[0],
            'nr of bins': combination_obj[1],
            'binning phase': combination_obj[2],
        }
    else:
        logger.error(
            f'More or less binning combination parameters in combination object than expected! (Expected 2 or 3, got {combo_len}) Now exiting.'
        )
        sys.exit()
