"""Python module containing custom functions that help reconstruct a label for a radial kernel, based on data loaded/generated from a RadialKernelData object.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import sys
import logging

# intra-package imports
from .radial_kernel_label_enumeration import (
    LabelInfoPlot,
    SpecificLabelInfoPlot,
)

# type checking imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # import the class needed to type this radial plot label generating function
    from .radial_kernel_helper import RadialKernelData


logger = logging.getLogger(__name__)


# RUN FUNCTION: create axis labels with correct formatting
def generate_plot_labels(radial_kernel_data: 'RadialKernelData') -> None:
    """Generates plot labels for the radial kernel terms in the 'radial_kernel_terms' dictionary.

    Parameters
    ----------
    radial_kernel_terms : dict
        Contains the necessary information to generate the plot labels for the radial kernels.
    """
    # loop over different terms
    for _val in radial_kernel_data.radial_kernel_terms.values():
        # convert mode legend labels
        if _val['symmetric']:
            legend_strings = [
                r', '.join(
                    [
                        LabelInfoPlot.get_plot_label(radial_kernel_name=_l)
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
                            radial_kernel_name=_label,
                            specifier=_label_specifier,
                        )
                        for _label, _label_specifier in zip(
                            _val['legend'], [r'A', r'B', r'C']
                        )
                    ]
                )
            ]
        # store legend entry
        _val['legend'] = [rf'${l}$' for l in legend_strings]  # noqa: E741
        # convert stellar multiplier labels
        try:
            stellar_multiplier_label = r'\,'.join(
                [
                    LabelInfoPlot.get_plot_label(radial_kernel_name=_mul)
                    for _mul in _val['s_e_multipliers']
                ]
            )
        except KeyError:
            logger.error(
                "'s_e_multiplier' was not a key of the RadialKernelDict, while it should be. Now exiting."
            )
            sys.exit()
        # generate legend title strings from those multipliers
        _val['legend_title'] = rf'${stellar_multiplier_label}$'
