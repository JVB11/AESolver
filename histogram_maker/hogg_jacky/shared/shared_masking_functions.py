"""Python module containing functions shared among different implementations for the jackknife likelihood computations that facilitate masking data.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import modules
import logging

# import type aliases from separate module
from ..typing_info import D_arr, B_arr


# set up a logger
logger = logging.getLogger(__name__)


def retrieve_valued_data(my_data: D_arr, nan_mask: B_arr) -> D_arr:
    """Retrieves a numpy array containing the non-NaN data for the binning.

    Parameters
    ----------
    my_data : D_arr
        Data array for which jackknife likelihood is to be computed.
    nan_mask : B_arr
        Boolean array containing the indices containing NaN data.

    Returns
    -------
    D_arr
        Numpy array containing non-NaN data.
    """
    return my_data[~nan_mask]
