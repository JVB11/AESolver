"""Python module containing generic kernel figure template function.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import AutoMinorLocator

# type checking imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib import figure
    from typing import Sequence
    from ...profile_helper_classes.radial_kernels import RadialKernelData
    import numpy as np
    import numpy.typing as npt


def _create_figure_gridspec(
    figsize: tuple[float | int, float | int],
    nr_plot_rows: int,
    nr_plot_cols: int,
) -> tuple['figure.Figure', GridSpec]:
    # generate figure
    fig = plt.figure(dpi=300, figsize=figsize)
    # generate gridspec
    gs = fig.add_gridspec(nr_plot_rows, nr_plot_cols, wspace=0, hspace=0)
    # return both
    return fig, gs


def _add_validity_domains(
    nr_plot_cols: int, nr_plot_rows: int, gridspec: GridSpec
) -> 'tuple[list[Axes], bool]':
    match (nr_plot_cols, nr_plot_rows):
        case (1, 1):
            # CREATE SINGLE AXIS
            my_sub = gridspec.subplots(sharex='col', sharey='row')
            if TYPE_CHECKING:
                assert isinstance(my_sub, Axes)
            # RETURN
            return [my_sub], True
        case (1, _) | (_, 1):
            # CREATE LIST/NDARRAY OF AXES
            ax = gridspec.subplots(sharex='col', sharey='row')
            if TYPE_CHECKING:
                assert isinstance(ax, list)
            # RETURN
            return ax, False
        case _:
            print(
                'This sort of plot grid is not (yet) implemented. Now exiting.'
            )
            import sys

            sys.exit()


def _adjust_tick_params(ax: 'Axes', single_ax: bool) -> None:
    # first adjustment: set labelsize; second adjustment add minor ticks and display on the relevant sides of the plot
    if single_ax:
        # first
        ax.tick_params(axis='both', labelsize=12)
        # second
        ax.tick_params(right=True, top=True, which='both')
    else:
        # first
        ax.tick_params(axis='both', labelsize=12, direction='in')
        # second
        ax.tick_params(right=True, top=True, which='both', direction='in')


def _display_outer_labels_only(ax_list: 'list[Axes]') -> None:
    for ax in ax_list:
        ax.label_outer()


def _set_x_axis_limits(ax: 'Axes', x_axis_limits: list[int | float]) -> None:
    ax.set_xlim(left=x_axis_limits[0], right=x_axis_limits[1])


def _set_axis_tick_locators(ax: 'Axes') -> None:
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())


def _set_axis_labels(
    nr_ax: int,
    ax: 'Axes',
    x_label: str,
    y_label_variable_part: str,
    ylabel_number: int,
) -> None:
    # set x-axis label
    ax.set_xlabel(x_label, size=14)
    # set y-axis label for first axis
    if nr_ax == 0:
        ax.set_ylabel(
            rf'$\left(\eta_1^{{({ylabel_number})}}\right)_{{{y_label_variable_part}}}$',
            size=14,
        )


def _add_tar_validity_domain_if_necessary(
    cc_contribution_data: 'None | RadialKernelData',
    tar_valid: 'Sequence[bool] | None | npt.NDArray[np.bool_]',
    ax: 'Axes',
) -> None:
    if (cc_contribution_data is not None) and (tar_valid is not None):
        ax.fill_between(
            cc_contribution_data.x,
            0,
            1,
            where=tar_valid,  # type: ignore (where also accepts a boolean Numpy array as input, in addition to a Sequence of booleans or None)
            transform=ax.get_xaxis_transform(),
            color='k',
            alpha=0.1,
            lw=0.0,
        )


def kernel_fig(
    ylabel_number: int,
    x_axis_limits: list[int | float],
    x_label: str,
    y_label_var_part: str,
    nr_plot_cols: int = 1,
    nr_plot_rows: int = 1,
    figsize: tuple[float, float] = (14.0, 12.0),
    cc_contribution_data: 'RadialKernelData | None' = None,
    tar_valid: 'Sequence[bool] | None | npt.NDArray[np.bool_]' = None,
) -> 'tuple[figure.Figure, Axes | list[Axes], GridSpec]':
    # generate figure and axes objects
    fig, my_gridspec = _create_figure_gridspec(
        figsize=figsize, nr_plot_cols=nr_plot_cols, nr_plot_rows=nr_plot_rows
    )
    # add validity domain(s)
    ax, single_ax = _add_validity_domains(
        nr_plot_cols=nr_plot_cols,
        nr_plot_rows=nr_plot_rows,
        gridspec=my_gridspec,
    )
    # perform formatting for these axes
    for _i, a in enumerate(ax):
        # add TAR validity domain, if necessary
        _add_tar_validity_domain_if_necessary(
            cc_contribution_data=cc_contribution_data, tar_valid=tar_valid, ax=a
        )
        # set axis limit
        _set_x_axis_limits(ax=a, x_axis_limits=x_axis_limits)
        # add tick locators
        _set_axis_tick_locators(ax=a)
        # set axis labels
        _set_axis_labels(
            nr_ax=_i,
            ax=a,
            x_label=x_label,
            y_label_variable_part=y_label_var_part,
            ylabel_number=ylabel_number,
        )
        # adjust tick parameters
        _adjust_tick_params(ax=a, single_ax=single_ax)
    # only show outer labels
    _display_outer_labels_only(ax_list=ax)
    # return figure, axes and gridspec
    return fig, ax, my_gridspec
