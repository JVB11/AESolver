"""Python module containing class that plots the quantities of the stellar evolution models (MESA).

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import sys
import logging
import matplotlib.pyplot as plt
from multimethod import multimethod

# relative imports
from .plotter import SuperAdNadComparisonPlotter as SanCP


# only show warning and higher level logs for matplotlib package
logging.getLogger('matplotlib').setLevel(level=logging.WARNING)
# set up logger
logger = logging.getLogger(__name__)


# class that contains the plotting-related methods for the corresponding stellar evolution model (MESA) computation results
class AdNadComparisonMESAPlotter(SanCP):
    """Python class containing the plotting-related methods for the corresponding stellar evolution model (MESA) computation results.

    Parameters
    ----------
    mesa : list[dict]
        Contains the MESA profile data dictionaries.
    """

    # type declarations
    _mesa: list[dict]

    # initialization method
    def __init__(self, mesa) -> None:
        # initialize superclass
        super().__init__()
        # store the gyre adiabatic and non-adiabatic results
        self._mesa = mesa

    # multimethod multi-profile plotter
    @multimethod
    def _multi_profile_plot(
        self,
        y_quantity: str,
        y_legend_label: str,
        y_normalizing_quantity,
        x_quantity,
        x_normalizing_quantity,
        dd,
        ax,
        qd,
    ):
        """Generic multi-method used to create multi-profile plots.

        Parameters
        ----------
        y_quantity : str or list
            Quantity or list of quantities plotted on the y-axis of the profile.
        y_legend_label : str or list or None
            The legend label for the y-axis quantity or a list of labels for the list of y-axis quantities.
        y_normalizing_quantity : str or None
            Normalizing quantity for the y-axis quantity (if not None).
        x_quantity : str
            Quantity plotted on the x-axis of the profile.
        x_normalizing_quantity : str or None
            Normalizing quantity for the x-axis quantity (if not None).
        dd : dict
            Holds the data to be accessed/plotted.
        qd : bool
            If True, and y_quantity is a list, and y_legend_label is None, compute the difference of the y-quantities and plot it.
        """
        # plot the single MESA profile with label
        ax.plot(
            self._normalizer(x_normalizing_quantity, x_quantity, dd),
            self._normalizer(y_normalizing_quantity, y_quantity, dd),
            label=y_legend_label,
        )

    @multimethod
    def _multi_profile_plot(
        self,
        y_quantity: str,
        y_legend_label: None,
        y_normalizing_quantity,
        x_quantity,
        x_normalizing_quantity,
        dd,
        ax,
        qd,
    ):
        # plot the single MESA profile without label
        ax.plot(
            self._normalizer(x_normalizing_quantity, x_quantity, dd),
            self._normalizer(y_normalizing_quantity, y_quantity, dd),
        )

    @multimethod
    def _multi_profile_plot(
        self,
        y_quantity: list,
        y_legend_label: list,
        y_normalizing_quantity,
        x_quantity,
        x_normalizing_quantity,
        dd,
        ax,
        qd,
    ):
        if (y_len := len(y_quantity)) == (y_leg_len := len(y_legend_label)):
            # plot the different MESA profiles on the same axis
            for _y_q, _y_l in zip(y_quantity, y_legend_label):
                self._multi_profile_plot(
                    _y_q,
                    _y_l,
                    y_normalizing_quantity,
                    x_quantity,
                    x_normalizing_quantity,
                    dd,
                    ax,
                    qd,
                )
        else:
            logger.error(
                f'Length of supplied lists does not match. (Length y-quantity list {y_len}, Length y-legend list {y_leg_len})'
            )

    @multimethod
    def _multi_profile_plot(
        self,
        y_quantity: str,
        y_legend_label: list,
        y_normalizing_quantity,
        x_quantity,
        x_normalizing_quantity,
        dd,
        ax,
        qd,
    ):
        logger.error(
            'Conflicting types: y-quantity = str; y-legend = list. Now exiting.'
        )
        sys.exit()

    @multimethod
    def _multi_profile_plot(
        self,
        y_quantity: list,
        y_legend_label: str,
        _normalizing_quantity,
        x_quantity,
        x_normalizing_quantity,
        dd,
        ax,
        qd,
    ):
        logger.error(
            'Conflicting types: y-quantity = list; y-legend = str. Now exiting.'
        )
        sys.exit()

    @multimethod
    def _multi_profile_plot(
        self,
        y_quantity: list,
        y_legend_label: None,
        y_normalizing_quantity,
        x_quantity,
        x_normalizing_quantity,
        dd,
        ax,
        qd,
    ):
        if qd:
            ax.plot(
                self._normalizer(x_normalizing_quantity, x_quantity, dd),
                dd[y_quantity[0]] - dd[y_quantity[1]],
            )
        else:
            logger.error(
                'Conflicting types: y-quantity = list; y-legend = None. (And we are not creating a quantity difference plot) Now exiting.'
            )
            sys.exit()

    # create a figure containing the relevant (MESA) profile
    def figure_profile(
        self,
        x_quantity,
        y_quantity,
        x_label,
        y_label,
        quantity_diff=False,
        x_bounds=None,
        y_bounds=None,
        mode_nr=1,
        y_legend_labels=None,
        x_normalizing_quantity=None,
        y_normalizing_quantity=None,
    ):
        """Generic method used to make a MESA profile figure.

        Parameters
        ----------
        x_quantity : str
            Quantity plotted on the x-axis of the profile.
        y_quantity : str or list
            Quantity or list of quantities plotted on the y-axis
            of the profile.
        x_label : str
            Label for the x-axis
        y_label : str
            Label for the y-axis
        quantity_diff : bool, optional

        x_bounds : None or Tuple or Dict, optional
            Defines the bounds for the x-axis. Default: None.
        y_bounds : None or Tuple or Dict, optional
            Defines the bounds for the y-axis. Do not use when providing multiple y-axis quantities; by default None.
        mode_nr : int, optional
            The mode number. Default: 1.
        y_legend_labels : list or None, optional
            Contains labels for the y-axis quantities. Default: None.
        x_normalizing_quantity: str or None, optional
            If a string, refers to quantity that is used to normalize the quantity to be plotted on the x-axis. Default: None.
        y_normalizing_quantity: str or None, optional
            If a string, refers to quantity that is used to normalize the quantity to be plotted on the y-axis. Default: None.
        """
        # create the figure
        _fig, _ax = plt.subplots()
        # get the data dictionary
        _dd = self._mesa[mode_nr]
        # plot the profile(s)
        self._multi_profile_plot(
            y_quantity,
            y_legend_labels,
            y_normalizing_quantity,
            x_quantity,
            x_normalizing_quantity,
            _dd,
            _ax,
            quantity_diff,
        )
        # perform common plotting actions
        self._common_actions_figures(
            ax=_ax,
            x_label=x_label,
            y_label=y_label,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
        )
        # set legend if necessary
        if y_legend_labels is not None:
            _ax.legend(title='MESA profile')
        # set tight layout
        _fig.tight_layout()
