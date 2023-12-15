"""Python module containing classes used to generate plot-ready labels for the different radial kernels.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
from enum import Enum


# define custom enumeration classes used to perform mapping from label information to plot format
class LabelInfoPlot(str, Enum):
    """Maps label information to strings used for plotting."""

    # generic enumeration member labels
    _Z_1 = r'z_1'
    _RAD_DIVERG = r'H'
    _RAD_DER_X_Z_1 = r'\dfrac{\partial(r\,z_1)}{\partial r}'
    _Z_2_OVER_C_1_OMEGA = r'\dfrac{z_2}{c_1\,\bar{\omega}^2}'
    _RAD_DER_X_Z_2_C_1_OMEGA = (
        r'\dfrac{\partial}{\partial r}\,(r\,\dfrac{z_2}{c_1\,\bar{\omega}^2})'
    )
    PRESSURE = r'P'
    _GAM_MIN = r'(\Gamma_1 - 1)'
    _GAM_MIN_SQ = r'(\Gamma_1 - 1)^2'
    DER_GAMMA_1_AD = (
        r'\left(\dfrac{\partial \Gamma_1}{\partial \ln \rho}\right)_S'
    )
    DENSITY = r'\rho'
    _R3 = r'r^3'
    _R2 = r'r^2'
    RADIAL_COORDINATE = r'r'
    D3GRAVPOT = r'\dfrac{\partial^3 \Phi}{\partial r^3}'
    D2GRAVPOT = r'\dfrac{\partial^2 \Phi}{\partial r^2}'
    DGRAVPOT = r'\dfrac{\partial \Phi}{\partial r}'

    # method used to retrieve the corresponding labels
    @classmethod
    def get_plot_label(cls, radial_kernel_name: str) -> str:
        """Retrieves the plot label for a radial kernel.

        Parameters
        ----------
        radial_kernel_name : str
            Name of the radial kernel.

        Returns
        -------
        str
            Plot label for the radial kernel.
        """
        # attempt to retrieve the radial kernel plot label
        try:
            return getattr(cls, radial_kernel_name.upper()).value
        except AttributeError:
            print(
                f'Unknown attribute requested ({radial_kernel_name}). Now returning empty string.'
            )
            return ''


class SpecificLabelInfoPlot(Enum):
    """Maps specific label information to strings used for plotting."""

    # non-symmetric, specific enumeration member labels
    _RAD_DIVERG = r'(H)_{{{spec}}}'
    _Z_1 = r'(z_{{1}})_{{{spec}}}'
    _Z_2_OVER_C_1_OMEGA = (
        r'\left(\dfrac{{z_2}}{{c_1\,\bar{{\omega}}^2}}\right)_{{{spec}}}'
    )
    _RAD_DER_X_Z_1 = (
        r'\left(\dfrac{{\partial(r\,z_1)}}{{\partial r}}\right)_{{{spec}}}'
    )

    # method used to retrieve the corresponding mode-specific labels
    @classmethod
    def get_plot_label(cls, radial_kernel_name: str, specifier: str) -> str:
        """Retrieves the plot label for a specific radial kernel.

        Parameters
        ----------
        radial_kernel_name : str
            Name of the radial kernel.
        specifier : str
            Specifier for the radial kernel.

        Returns
        -------
        str
            Plot label for the specific radial kernel.
        """
        # attempt to retrieve the radial kernel plot label
        try:
            return getattr(cls, radial_kernel_name.upper()).value.format(
                spec=specifier
            )
        except AttributeError:
            print(
                f'Unknown attribute requested ({radial_kernel_name}, specifier={specifier}). Now returning empty string.'
            )
            return ''
