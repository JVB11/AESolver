"""Python module containing functions shared among different implementations for the jackknife likelihood computations that facilitate printing information about the optimization result to the stdout stream.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import modules
import logging
import numpy as np

# import type aliases from separate module
from ..typing_info import D_var, I_var, D_arr


# set up a logger
logger = logging.getLogger(__name__)


def print_single_value(
    custom_bin_width: D_var | None,
    ones_combo: tuple[D_var, I_var] | None,
    max_bin_phase: D_var | None = None,
) -> None:
    """Prints the values of the search grid when there is only 1 data point available.

    Parameters
    ----------
    custom_bin_width : D_var
        The custom bin width defined by the user to plot a single data point.
    ones_combo : tuple[D_var, I_var]
        The default alpha value (1.0) and number of bins (1) for a single data point.
    """
    if ones_combo is None:
        print(
            f'The likelihood value was not computed because there is only 1 data point available! We used a custom bin width with a value of {custom_bin_width}, a default alpha parameter with a value of {ones_combo} and a single bin (number of bins = {ones_combo}).'
        )
    else:
        print(
            f'The likelihood value was not computed because there is only 1 data point available! We used a custom bin width with a value of {custom_bin_width}, a default alpha parameter with a value of {ones_combo[0]} and a single bin (number of bins = {ones_combo[1]}).'
        )


def print_maximal_values(
    max_bin_width_edges: D_var | D_arr,
    max_combo: tuple[D_var, I_var],
    max_bin_phase: D_var | None = None,
) -> None:
    """Prints the values of the search grid for which the jackknife likelihood is maximized.

    Parameters
    ----------
    max_bin_width : D_var | D_arr
        The bin width for which the jackknife likelihood is maximized, or the bin edges for which the jackknife likelihood is maximized (log bins).
    max_combo : tuple[D_var, I_var]
        The alpha value and number of bins for which the jackknife likelihood is maximized.
    max_bin_phase : D_var | None
        If None, this value was not used during binning (numpy histogram method), and will thus be ignored during print statements. Otherwise, this represents the binning phase factor defined in Hogg(2008) for the parameter triad that maximizes the jackknife likelihood; by default None.
    """
    if isinstance(max_bin_width_edges, np.ndarray):
        if max_bin_phase:
            print(
                f'The likelihood value was maximized for bin edges {max_bin_width_edges}, an alpha parameter with a value of {max_combo[0]:.1f} and {max_combo[1]} bins. The binning phase for this triad is {max_bin_phase}.'
            )
        else:
            print(
                f'The likelihood value was maximized for bin edges {max_bin_width_edges}, an alpha parameter with a value of {max_combo[0]:.1f} and {max_combo[1]} bins.'
            )
    else:
        if max_bin_phase:
            print(
                f'The likelihood value was maximized for a bin width of {max_bin_width_edges}, an alpha parameter with a value of {max_combo[0]:.1f} and {max_combo[1]} bins. The binning phase for this triad is {max_bin_phase}.'
            )
        else:
            print(
                f'The likelihood value was maximized for a bin width of {max_bin_width_edges}, an alpha parameter with a value of {max_combo[0]:.1f} and {max_combo[1]} bins.'
            )
