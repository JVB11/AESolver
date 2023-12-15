"""Python module containing functions shared among different implementations for the jackknife likelihood computations that facilitate checking assumptions about the provided data.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import modules
import logging

# import type aliases from separate module
from ..typing_info import D_arr, D_list, I_arr, I_list


# set up a logger
logger = logging.getLogger(__name__)


# check input data
def check_input_data(
    my_data: D_arr,
    my_alpha_parameters: D_arr | D_list,
    my_bin_numbers: I_arr | I_list,
) -> bool:
    """Checks whether the input data is valid to compute the jackknife likelihood used to determine bin widths of histograms, following Hogg (2008).

    Parameters
    ----------
    my_data : D_arr
        Data array for which jackknife likelihood is to be computed or None.
    my_alpha_parameters : D_arr | D_list
        Contains the alpha smoothing parameters for the Hogg(2008) jackknife likelihood (or None).
    my_bin_numbers : I_arr | I_list
        Contains several numbers of bins to be tried when computing the jackknife likelihood (or None).

    Returns
    -------
    bool
        True if the input data are OK to be used for jackknife likelihood computation to determine bin widths following Hogg (2008).
    """
    if any(
        [my_data is None, my_alpha_parameters is None, my_bin_numbers is None]
    ):
        # not all available: log an error
        logger.error(
            'Not all necessary data available to estimate the Hogg (2008) jackknife likelihood. Will now exit!'
        )
        return False
    else:
        return True


# check data shape
def check_data_shape(my_data: D_arr) -> bool:
    """Checks if the data shape is OK to compute the jackknife likelihood.

    Parameters
    ----------
    my_data : np.ndarray
        Data array for which jackknife likelihood is to be computed.

    Returns
    -------
    bool
        True if data shape is OK for the likelihood computation, False if not.
    """
    return my_data.shape[0] != 0
