"""Python module containing classes used to generate plot-ready labels for the different angular kernels.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# TODO: move this to plotting, as it should not be part of solver [FEATURE UPDATE]
# import statements
from enum import Enum


# define custom enumeration classes used to perform mapping from label information to plot format
class LabelInfoPlot(str, Enum):
    """Maps label information to strings used for plotting."""

    # generic enumeration member labels
    HR = r'H_r'
    HT = r'H_\theta'
    HP = r'H_\phi'
    MHR = r'm\,H_r'
    THETA_DER_HR_THETA = r'\partial\,H_r'
    THETA_DER_HT_THETA = r'\partial\,H_\theta'
    THETA_DER_HP_THETA = r'\partial\,H_\phi'
    PHI_PHI = r'H_\phi^\phi'
    THETA_PHI = r'H_\phi^\theta'

    # method used to retrieve the corresponding labels
    @classmethod
    def get_plot_label(cls, angular_kernel_name: str) -> str:
        """Retrieves the plot label for an angular kernel.

        Parameters
        ----------
        angular_kernel_name : str
            Name of the angular kernel.

        Returns
        -------
        str
            Plot label for the angular kernel.
        """
        # attempt to retrieve the angular kernel plot label
        try:
            return getattr(cls, angular_kernel_name.upper()).value
        except AttributeError:
            print(
                f'Unknown attribute requested ({angular_kernel_name}). Now returning empty string.'
            )
            return ''


class SpecificLabelInfoPlot(Enum):
    """Maps specific label information to strings used for plotting."""

    # non-symmetric, specific enumeration member labels
    HR = r'\left(H_r\right)_{{{spec}}}'
    HT = r'\left((H_\theta\right)_{{{spec}}}'
    HP = r'\left(H_\phi\right)_{{{spec}}}'
    MHR = r'\left(m\,H_r\right)_{{{spec}}}'
    THETA_DER_HR_THETA = r'\left(\partial\,H_r\right)_{{{spec}}}'
    THETA_DER_HT_THETA = r'\left(\partial\,H_\theta\right)_{{{spec}}}'
    THETA_DER_HP_THETA = r'\left(\partial\,H_\phi\right)_{{{spec}}}'
    PHI_PHI = r'\left(H_\phi^\phi\right)_{{{spec}}}'
    THETA_PHI = r'\left(H_\phi^\theta\right)_{{{spec}}}'

    # method used to retrieve the corresponding mode-specific labels
    @classmethod
    def get_plot_label(cls, angular_kernel_name: str, specifier: str) -> str:
        """Retrieves the plot label for a specific angular kernel.

        Parameters
        ----------
        angular_kernel_name : str
            Name of the angular kernel.
        specifier : str
            Specifier for the angular kernel.

        Returns
        -------
        str
            Plot label for the specific angular kernel.
        """
        # attempt to retrieve the angular kernel plot label
        try:
            return getattr(cls, angular_kernel_name.upper()).value.format(
                spec=specifier
            )
        except AttributeError:
            print(
                f'Unknown attribute requested ({angular_kernel_name}, specifier={specifier}). Now returning empty string.'
            )
            return ''
