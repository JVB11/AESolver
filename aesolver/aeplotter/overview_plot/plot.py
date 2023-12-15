"""Module containing plotting functions used to generate the overview plot.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
from collections.abc import Sequence
from matplotlib.ticker import AutoMinorLocator, FuncFormatter, LinearLocator
from ..data_files import OverviewPlotOptions

# type checking imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    from matplotlib import axes
    from typing import TypedDict
    from .figure_class import OverviewFigure
    from ..data_files.overview_plot_data import PlotOptions

    class PlotKwargs(TypedDict):
        c: str
        s: int
        marker: str
        label: str
        zorder: int
        alpha: float


# frequency-period conversion function
def freq_period_conversion(
    f_or_p: 'float | npt.NDArray[np.float64]'
) -> 'float | npt.NDArray[np.float64]':
    return 1.0 / f_or_p


# retrieve actual plotting kwargs
def _get_plotting_kwargs(
    my_category: str, multiplication_factor_alpha: float = 1.0
) -> 'PlotKwargs':
    # retrieve the plot options for your chosen category
    plot_options: 'PlotOptions | None' = (
        OverviewPlotOptions().get_dict().get(my_category)
    )
    # return the plotting kwargs, if an attribute is returned
    if plot_options is None:
        raise NotImplementedError(
            'The specific category {my_category} is not a valid plot option (valid options are: {}). Please select a different plot option when generating the overview plot.'
        )
    else:
        return {
            'c': plot_options['c'],
            's': 1,
            'marker': '.',
            'label': plot_options['l'],
            'zorder': plot_options['z'],
            'alpha': multiplication_factor_alpha * plot_options['a'],
        }


# function used to plot the predicted data on the target figure
def plot_predicted_data_on_target_figure(
    figure_object: 'OverviewFigure'
) -> None:
    # get the different categories
    _categories = OverviewPlotOptions().get_plot_categories()
    # unpack the data from the figure object
    x = figure_object._min_freq_stack
    y = figure_object._amp_ratio_stack
    stable_mask = figure_object._mask_stable
    ae_stable_mask = figure_object._mask_ae_stable
    stable_valid_mask = figure_object._mask_stable_valid
    neg_mask = figure_object._negative_value_mask
    my_ax = figure_object._ax
    # verify you can plot (i.e. if my_ax is not None)
    if my_ax is not None:
        # loop over the different masks to plot the data
        for _cat, _mask in zip(
            _categories, [stable_mask, ae_stable_mask, stable_valid_mask]
        ):
            # store masked x and y values
            _masked_x = x[_mask]
            _masked_y = y[_mask]
            # get the negative and positive masks --> for styling
            _my_neg_mask = neg_mask[_mask]
            _my_pos_mask = ~_my_neg_mask
            # plot the data
            # - regular data
            my_ax.scatter(
                _masked_x[_my_pos_mask],
                _masked_y[_my_pos_mask],
                **_get_plotting_kwargs(
                    my_category=_cat, multiplication_factor_alpha=1.0
                ),
            )
            # - mirrored data
            my_ax.scatter(
                _masked_x[_my_neg_mask],
                _masked_y[_my_neg_mask],
                **_get_plotting_kwargs(
                    my_category=_cat, multiplication_factor_alpha=0.005
                ),
            )


# FORMATTING OPTIONS
# set axis labels
def _set_axis_labels_theoretical(
    my_axis: 'axes.Axes', my_second_axis: 'axes.Axes'
) -> None:
    # provide labels for main axis
    my_axis.set_xlabel(
        r'$\Omega^{\rm NL,s}_{{\rm min},\,\mathfrak{i}}\,/\,2\pi$ [d$^{-1}$]',
        fontsize=13,
    )
    my_axis.yaxis.set_label_position('right')
    my_axis.set_ylabel(
        r'$|\mathfrak{L}_{\rm d}|\,/\,|\mathfrak{L}_{\rm p}|$',
        fontsize=13,
        rotation=-90,
        labelpad=15,
    )
    # provide labels for secondary axis
    my_second_axis.set_xlabel(r'Period [d]', fontsize=13)


# set legend
def _create_legend_for_figure(my_axis: 'axes.Axes') -> None:
    # obtain handles and labels
    handles, labels = my_axis.get_legend_handles_labels()
    # retrieve the unique legend labels and handles
    unique_legend_labels_handles = [
        (h, l)
        for i, (h, l) in enumerate(zip(handles, labels))  # noqa: E741
        if l not in labels[:i]
    ]
    # create the legend (without a title)
    my_axis.legend(
        *zip(*unique_legend_labels_handles), shadow=True, fontsize=11
    )


def _set_y_axis_log_scale(my_axis: 'axes.Axes') -> None:
    # set y-axis to log scale
    my_axis.set_yscale('log')


def _set_axis_limits(my_axis: 'axes.Axes') -> None:
    # x limit
    my_axis.set_xlim(left=0.0001, right=5.999)
    # y limit
    my_axis.set_ylim(bottom=0.001, top=1000.0)


class MyLoc(LinearLocator):
    def tick_values(self, vmin: float, vmax: float) -> Sequence[float]:
        return super().tick_values(vmin, vmax)


def _set_locators_formatters_axis(
    my_axis: 'axes.Axes', my_second_axis: 'axes.Axes'
) -> None:
    # convert to frequency to period on secondary axis
    my_second_axis.xaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: f'{freq_period_conversion(x):.2f}')
    )
    # set minor locators on both sets of axes
    my_axis.xaxis.set_minor_locator(AutoMinorLocator())


def _set_tick_parameters_axis(
    my_axis: 'axes.Axes', my_second_axis: 'axes.Axes'
) -> None:
    # set tick parameters
    my_axis.tick_params(
        which='both',
        axis='y',
        left=True,
        right=True,
        labelsize=12,
        labelleft=False,
        labelright=True,
    )
    my_axis.tick_params(which='both', axis='x', labelsize=12)
    # tick parameters second axis
    my_second_axis.tick_params(which='both', axis='x', labelsize=12)


def format_overview_plot(my_axis: 'axes.Axes', my_second_axis: 'axes.Axes'):
    # set axis log scale
    _set_y_axis_log_scale(my_axis=my_axis)
    # set tick parameters
    _set_tick_parameters_axis(my_axis=my_axis, my_second_axis=my_second_axis)
    # set locators
    _set_locators_formatters_axis(
        my_axis=my_axis, my_second_axis=my_second_axis
    )
    # set axis limits
    _set_axis_limits(my_axis=my_axis)
    # set axis labels
    _set_axis_labels_theoretical(my_axis=my_axis, my_second_axis=my_second_axis)
    # create legend
    _create_legend_for_figure(my_axis=my_axis)
