"""Python module containing custom functions that help reconstruct a label for an angular kernel, based on data loaded/generated from an AngularKernelData object.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import math
import sys
import logging

# intra-package imports
from .angular_kernel_label_enumeration import (
    LabelInfoPlot,
    SpecificLabelInfoPlot,
)

# type checking imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # import the class needed to type this angular plot label generating function
    from .angular_kernel_helper import AngularKernelData


logger = logging.getLogger(__name__)


# RUN FUNCTION: create axis labels with correct formatting
def generate_angular_plot_labels(
    angular_kernel_data: 'AngularKernelData'
) -> None:
    """Store angular kernel plot labels in 'angular_kernel_terms', based on data already stored in that dictionary.

    Parameters
    ----------
    angular_kernel_terms : dict
        Contains the necessary data to generate the angular kernel plot labels.
    """
    # loop over different terms
    for _val in angular_kernel_data.angular_kernel_terms.values():
        # convert mode legend labels
        if _val['symmetric']:
            # SYMMETRIZED TERM
            legend_strings = [
                r', '.join(
                    [
                        LabelInfoPlot.get_plot_label(angular_kernel_name=_l)
                        for _l in _label_tuple
                    ]
                )
                for _label_tuple in _val['legend']
            ]
        else:
            legend_strings = [
                r', '.join(
                    [
                        SpecificLabelInfoPlot.get_plot_label(
                            angular_kernel_name=_label,
                            specifier=_label_specifier,
                        )
                        for _label, _label_specifier in zip(
                            _val['legend'], [r'A', r'B', r'C']
                        )
                    ]
                )
            ]
        # store legend entry in the provided input dictionary, with the corresponding multiplier
        try:
            _val['legend_string'] = _get_total_label_string(
                mul_with_mu=_val['mul_with_mu'],
                div_by_sin=_val['div_by_sin'],
                multiplying_factor=_val['multiplying_factor'],
                partial_strings=legend_strings,
                symmetric=_val['symmetric'],
            )
        except KeyError:
            logger.exception(
                "At least one of the keys 'mul_with_mu', 'div_by_sin', and 'multiplying_factor' is missing from the AngularKernelDict, while they should be valid keys. Now exiting."
            )
            sys.exit()


def _get_total_label_string(
    mul_with_mu: bool,
    div_by_sin: bool,
    multiplying_factor: float,
    partial_strings: list[str],
    symmetric: bool,
) -> list[str]:
    """Retrieves the plot label strings for plotting purposes of the different angular kernels.

    Parameters
    ----------
    mul_with_mu : bool
        If True, multiply the product kernel with cosine(theta) = Mu.
    div_by_sin : bool
        If True, divide the product kernel by sine(theta).
    multiplying_factor : float
        Multiply the product kernel with this factor. Set to 1.0 if no multiplication is necessary.
    partial_strings : list[str]
        Partial labels for the angular kernels.
    symmetric : bool
        True if a symmetrized expression is used, False if not.

    Returns
    -------
    list[str]
        Contains the plot labels for each of the different loaded angular kernels.
    """
    return [
        _get_individual_label_string(
            mul_with_mu=mul_with_mu,
            div_by_sin=div_by_sin,
            multiplying_factor=multiplying_factor,
            partial_string=_str,
            symmetric=symmetric,
        )
        for _str in partial_strings
    ]


def _get_individual_label_string(
    mul_with_mu: bool,
    div_by_sin: bool,
    multiplying_factor: float,
    partial_string: str,
    symmetric: bool = True,
) -> str:
    """Creates a plot label string for an individual angular kernel.

    Parameters
    ----------
    mul_with_mu : bool
        If True, multiply the product kernel with cosine(theta) = Mu.
    div_by_sin : bool
        If True, divide the product kernel by sine(theta).
    multiplying_factor : float
        Multiply the product kernel with this factor. Set to 1.0 if no multiplication is necessary.
    partial_string : str
        Partial label for the angular kernel.
    symmetric : bool, optional
        True if a symmetrized expression is used, False if not; by default True.

    Returns
    -------
    str
        Plot label string for the angular kernel.
    """
    if symmetric:
        my_partial_string = rf'[{partial_string}]'
    else:
        my_partial_string = partial_string
    match (mul_with_mu, div_by_sin):
        case (False, False):
            if _is_close_1(multiplying_factor):
                return rf'${my_partial_string}$'
            elif _is_close_m_1(multiplying_factor):
                return rf'$-\,{my_partial_string}$'
            else:
                return rf'${multiplying_factor}\,[{my_partial_string}]$'
        case (True, False):
            if _is_close_1(multiplying_factor):
                return rf'$\mu\,{my_partial_string}$'
            elif _is_close_m_1(multiplying_factor):
                return rf'$-\,\mu\,{my_partial_string}$'
            else:
                joined_string = r'\,'.join([rf'{multiplying_factor}', r'\mu'])
                return rf'${joined_string}\,{my_partial_string}$'
        case (False, True):
            if _is_close_1(multiplying_factor):
                numerator_string = r''
            elif _is_close_m_1(multiplying_factor):
                numerator_string = r'-\,'
            else:
                numerator_string = rf'{multiplying_factor}\,'
            joined_string = rf'{numerator_string}{my_partial_string}'
            return rf'$\dfrac{{{joined_string}}}{{\sqrt{{1 - \mu^2}}}}$'
        case (True, True):
            if _is_close_1(multiplying_factor):
                numerator_string = r'\mu\,'
            elif _is_close_m_1(multiplying_factor):
                numerator_string = r'-\,\mu\,'
            else:
                numerator_string = rf'{multiplying_factor}\,\mu\,'
            joined_string = rf'{numerator_string}{my_partial_string}'
            return rf'$\dfrac{{{joined_string}}}{{\sqrt{{1 - \mu^2}}}}$'
        case _:
            raise NotImplementedError(
                "This feature was not implemented. Make sure 'mul_with_mu' and 'div_by_sin' are boolean values!"
            )


def _is_close_1(val: float) -> bool:
    """Verifies if 'val' is close to 1 for floating point numbers.

    Parameters
    ----------
    val : float
        The floating point number for which equality to 1 must be checked.

    Returns
    -------
    bool
        True if 'val' is close to 1. False otherwise.
    """
    return math.isclose(val, 1.0)


def _is_close_m_1(val: float) -> bool:
    """Verifies if 'val' is close to -1 for floating point numbers.

    Parameters
    ----------
    val : float
        The floating point number for which equality to -1 must be checked.

    Returns
    -------
    bool
        True if 'val' is close to -1. False otherwise.
    """
    return math.isclose(val, -1.0)
