"""Python module containing superclass that has common methods for the separate plotting modules.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import logging
from functools import singledispatchmethod
from matplotlib.ticker import AutoMinorLocator


# only show warning and higher level logs for matplotlib package
logging.getLogger('matplotlib').setLevel(level=logging.WARNING)
# set up logger
logger = logging.getLogger(__name__)


# superclass that contains common plotting-related methods for the comparison of non-adiabatic with adiabatic stellar pulsation code (GYRE) and the corresponding stellar evolution model (MESA) computation results
class SuperAdNadComparisonPlotter:
    """Python superclass containing common methods used for plotting comparison plots for comparison of adiabatic and non-adiabatic stellar pulsation code (GYRE) and stellar evolution code (MESA) computation results."""

    # dummy initialization method
    def __init__(self) -> None:
        pass

    # bounds-setting method
    @singledispatchmethod
    def _bounds_setter(self, bounds, ax, x_b=True):
        """Generic method used to set specific bounds.

        Parameters
        ----------
        bounds : None or Tuple or Dict, optional
            Defines the bounds for a specific axis defined by 'x_b'.
        ax : plt.axes
            Matplotlib axes object for which boundaries need to be set.
        x_b : bool, optional
            If True, set the x-bounds, if False, set the y-bounds; by default True.
        """
        raise NotImplementedError("Could not determine the type of 'bounds'.")

    @_bounds_setter.register
    def _(self, bounds: tuple, ax, x_b=True):
        if x_b:
            ax.set_xlim(bounds)
        else:
            ax.set_ylim(bounds)

    @_bounds_setter.register
    def _(self, bounds: dict, ax, x_b=True):
        if x_b:
            ax.set_xlim(**bounds)
        else:
            ax.set_ylim(**bounds)

    @_bounds_setter.register
    def _(self, bounds: None, ax, x_b=True):
        pass

    # normalizing method
    @singledispatchmethod
    def _normalizer(self, normalizing_quantity, quantity, data_dict):
        """Generic method used to normalize a quantity.

        Parameters
        ----------
        normalizing_quantity : str or None or Tuple
            Points to the quantity used to normalize. No action undertaken if None.
        quantity : str
            Points to the quantity to be normalized.
        data_dict : dict
            The dictionary containing the data.

        Returns
        -------
        np.ndarray
            The normalized quantity. (Normalized if 'normalizing_quantity' is not None)
        """
        raise NotImplementedError(
            "Cannot determine the type of 'normalizing_quantity'."
        )

    @_normalizer.register
    def _(self, normalizing_quantity: None, quantity, data_dict):
        return data_dict[quantity]

    @_normalizer.register
    def _(self, normalizing_quantity: str, quantity, data_dict):
        return data_dict[quantity] / data_dict[normalizing_quantity]

    @_normalizer.register
    def _(self, normalizing_quantity: tuple, quantity, data_dict):
        return data_dict[quantity] / normalizing_quantity[0](
            data_dict[normalizing_quantity[1]]
        )

    # method that performs common actions in creation of figures
    def _common_actions_figures(
        self,
        ax,
        x_label,
        y_label,
        x_bounds,
        y_bounds,
        use_x_label=True,
        use_y_label=True,
    ):
        """Generic method used to perform common actions for creation of figures.

        Parameters
        ----------
        ax : plt.axes
            The Matplotlib axes object on which the common plotting actions are performed.
        x_label : str
            Label for the x-axis.
        y_label : str
            Label for the y-axis.
        x_bounds : None or Tuple or Dict
            Defines the bounds for the x-axis.
        y_bounds : None or Tuple or Dict
            Defines the bounds for the y-axis.
        use_x_label : bool, optional
            If True, plot an x-axis label. If False, do not; by default True.
        use_y_label : bool, optional
            If True, plot an y-axis label. If False, do not; by default True.
        """
        # set the axis labels
        if use_x_label:
            ax.set_xlabel(x_label, fontsize=13)
        if use_y_label:
            if isinstance(y_label, str):
                ax.set_ylabel(y_label, fontsize=13)
            else:
                logger.error(
                    f'Y-axis label was of wrong type ({type(y_label)}). Will not set y-axis label.'
                )
        # set tick parameters
        ax.tick_params(top=True, right=True, which='both', labelsize=11)
        ax.tick_params(axis='both', size=6)
        # set minor locators
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        # set axis bounds
        self._bounds_setter(x_bounds, ax, x_b=True)  # x-bounds
        self._bounds_setter(y_bounds, ax, x_b=False)  # y-bounds
