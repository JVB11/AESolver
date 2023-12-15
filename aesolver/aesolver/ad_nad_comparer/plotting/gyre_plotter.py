"""Python module containing class that plots the comparisons of the adiabatic and non-adiabatic stellar pulsation computations (GYRE).

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import logging
import sys
import matplotlib.pyplot as plt
from collections.abc import ByteString
from functools import singledispatchmethod

# relative imports
from .plotter import SuperAdNadComparisonPlotter as SanCP

# custom package imports
from gyre_unit_converter import GYREFreqUnitConverter


# only show warning and higher level logs for matplotlib package
logging.getLogger('matplotlib').setLevel(level=logging.WARNING)
# set up logger
logger = logging.getLogger(__name__)


# class that contains the plotting-related methods for the comparison of non-adiabatic with adiabatic stellar pulsation code (GYRE) results
class AdNadComparisonGYREPlotter(SanCP):
    """Python class containing plotting-related methods for the comparison of non-adiabatic with adiabatic stellar pulsation code (GYRE) results.

    Parameters
    ----------
    gyre_ad : list
        Contains the adiabatic GYRE results.
    gyre_nad : list
        Contains the non-adiabatic GYRE results.
    """

    # type declarations
    _gyre_ad: list
    _gyre_nad: list

    # initialization method
    def __init__(self, gyre_ad, gyre_nad) -> None:
        # initialize the superclass
        super().__init__()
        # store the gyre adiabatic and non-adiabatic results
        self._gyre_ad = gyre_ad
        self._gyre_nad = gyre_nad
        # compute the mass profiles
        self._compute_mass_profiles()

    # compute the mass profiles
    def _compute_mass_profiles(self):
        """Internal utility method that computes the mass profile for use in GYRE comparison plots."""
        # compute and add the mass profiles
        for _x, _xn in zip(self._gyre_ad, self._gyre_nad):
            _x['mass'] = _x['M_r'] / _x['M_star']
            _xn['mass'] = _xn['M_r'] / _xn['M_star']

    # quantity getting method
    @staticmethod
    def _quantity_getter(quant_dict, quantity, real_quantity):
        """Method used to obtain the quantity from one of the GYRE detail dictionaries.

        Parameters
        ----------
        quant_dict : dict
            Contains the values for the specific mode.
        quantity : str
            The quantity to be obtained.
        real_quantity : bool or None
            Real (TRUE) or imaginary (FALSE) part of quantity, or just its value (NONE).
        """
        try:
            if real_quantity is None:
                return quant_dict[quantity]
            elif real_quantity:
                return quant_dict[quantity]['re']
            else:
                return quant_dict[quantity]['im']
        except AttributeError:
            logger.exception(
                'Wrong attribute passed to quantity dictionary. Now exiting.'
            )
            sys.exit()

    # create a single plot on the figure
    def _single_plot_fig_comp(
        self,
        x_quantity,
        y_quantity,
        real_quantity_x,
        real_quantity_y,
        x_label,
        y_label,
        mode_nr,
        x_bounds,
        y_bounds,
        ax,
        use_title=True,
        use_x_label=True,
        use_legend=True,
    ):
        """Create figure containing a single plot that compares GYRE adiabatic and non-adiabatic quantities.

        Parameters
        ----------
        x_quantity : str
            Quantity to be displayed on the x-axis.
        y_quantity : str
            Quantity to be displayed on the y-axis.
        real_quantity_x : bool or None
            Real (TRUE) or imaginary (FALSE) part of quantity to be displayed on the x-axis, or just its value (NONE).
        real_quantity_y : bool or None
            Real (TRUE) or imaginary (FALSE) part of quantity to be displayed on the y-axis, or just its value (NONE).
        x_label : str
            Label for the x-axis.
        y_label : str
            Label for the y-axis.
        mode_nr : int
            The nr. of the mode selected in the mode triplet. Options are: [1,2,3].
        x_bounds : None or Tuple or Dict
            Defines the bounds for the x-axis. None enforces no bounds.
        y_bounds : None or Tuple or Dict
            Defines the bounds for the y-axis. None enforces no bounds.
        ax : matplotlib.Axes
            The matplotlib axes.
        use_title : bool, optional
            If True, set a title for the plot. If False, do not set such title; by default True.
        use_legend : bool, optional
            If True, plot a legend. If False, do not; by default True.
        """
        # get the data dictionaries for the specific mode
        _ad_mode = self._gyre_ad[mode_nr - 1]
        _nad_mode = self._gyre_nad[mode_nr - 1]
        # plot the adiabatic and non-adiabatic quantities
        ax.plot(
            self._quantity_getter(_ad_mode, x_quantity, real_quantity_x),
            self._quantity_getter(_ad_mode, y_quantity, real_quantity_y),
            label='adiabatic',
        )  # adiabatic quantity plot
        ax.plot(
            self._quantity_getter(_nad_mode, x_quantity, real_quantity_x),
            self._quantity_getter(_nad_mode, y_quantity, real_quantity_y),
            label='non-adiabatic',
        )  # non-adiabatic quantity plot
        # perform common plotting actions
        self._common_actions_figures(
            ax=ax,
            x_label=x_label,
            y_label=y_label,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            use_x_label=use_x_label,
        )
        # add the legend
        if use_legend:
            ax.legend(ncol=2, frameon=False)
        # add a descriptive title containing mode information
        if use_title:
            ax.set_title(
                rf"Mode (l,m,n): ({_ad_mode['l']},{_ad_mode['m']},{_ad_mode['n_g']})"
                + '\n'
                + rf"ad/nad $\nu$: {_ad_mode['freq']}/{_nad_mode['freq']} {GYREFreqUnitConverter.convert(_ad_mode['freq_units'])}"
                + '\n'
                + rf"({self._byte_string_converter(_ad_mode['freq_frame']).lower()} frame)",
                fontsize=13,
                pad=10,
            )

    # create a figure comparing (GYRE) adiabatic and non-adiabatic quantities
    def figure_comparison(
        self,
        x_quantity='x',
        y_quantity='xi_r',
        real_quantity_x=None,
        real_quantity_y=True,
        x_label=r'r / R$_*$',
        y_label='',
        mode_nr=1,
        x_bounds=None,
        y_bounds=None,
    ):
        """Generic method used to make a comparison figure, in which the GYRE adiabatic and non-adiabatic quantities are compared.

        Parameters
        ----------
        x_quantity : str, optional
            Quantity to be displayed on the x-axis. Default: 'x'.
        y_quantity : str, optional
            Quantity to be displayed on the y-axis. Default 'xi_r'.
        real_quantity_x : bool or None, optional
            Real (TRUE) or imaginary (FALSE) part of quantity to be displayed on the x-axis, or just its value (NONE); by default None.
        real_quantity_y : bool or None, optional
            Real (TRUE) or imaginary (FALSE) part of quantity to be displayed on the y-axis, or just its value (NONE); by default None.
        x_label : str, optional
            Label for the x-axis. Default: r'r / R$_*$'.
        y_label : str or list[str], optional
            Label for the y-axis or labels for y-axes. Default: ''.
        mode_nr : int, optional
            The nr. of the mode selected in the mode triplet. Options are: [1,2,3]. Default: 1.
        x_bounds : None or Tuple or Dict, optional
            Defines the bounds for the x-axis. None enforces no bounds; by default None.
        y_bounds : None or Tuple or Dict, optional
            Defines the bounds for the y-axis. None enforces no bounds; by default None.
        """
        # Determine whether multiple subplots need to be made, or only 1 single plot. (This choice is made based on the supplied y-axis label(s).)
        if (isinstance(y_label, list) or isinstance(y_label, tuple)) and (
            isinstance(real_quantity_y, list)
            or isinstance(real_quantity_y, tuple)
        ):
            if (length_labels := len(y_label)) == 2 and (
                length_reals := len(real_quantity_y) == 2
            ):
                # create the figure containing the double axes object
                _fig, (_ax1, _ax2) = plt.subplots(
                    nrows=2, ncols=1, sharex=True, dpi=200
                )
                # plot the data
                self._single_plot_fig_comp(
                    x_quantity,
                    y_quantity,
                    real_quantity_x,
                    real_quantity_y[0],
                    x_label,
                    y_label[0],
                    mode_nr,
                    x_bounds,
                    y_bounds,
                    _ax1,
                    use_x_label=False,
                )
                self._single_plot_fig_comp(
                    x_quantity,
                    y_quantity,
                    real_quantity_x,
                    real_quantity_y[1],
                    x_label,
                    y_label[1],
                    mode_nr,
                    x_bounds,
                    y_bounds,
                    _ax2,
                    use_title=False,
                    use_legend=False,
                )
            else:
                logger.error(
                    f'Wrong size of label list ({length_labels}) or wrong size of boolean list ({length_reals}). Both should be of length 2.'
                )
                sys.exit()
        else:
            # create the figure containing a single axes object
            _fig, _ax = plt.subplots()
            # plot the data
            self._single_plot_fig_comp(
                x_quantity,
                y_quantity,
                real_quantity_x,
                real_quantity_y,
                x_label,
                y_label,
                mode_nr,
                x_bounds,
                y_bounds,
                _ax,
            )
        # set tight layout
        _fig.tight_layout()

    @singledispatchmethod
    def _byte_string_converter(self, mode_string):
        """Converts a possible byte-string into a normal string.

        Parameters
        ----------
        mode_string : str or ByteString
            The string to be converted if it is a ByteString.

        Returns
        -------
        str
            The converted string.
        """
        raise NotImplementedError(
            "The type of 'mode_string' was not specified."
        )

    @_byte_string_converter.register
    def _(self, mode_string: str):
        return mode_string

    @_byte_string_converter.register
    def _(self, mode_string: ByteString):
        return mode_string.decode()
