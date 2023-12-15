"""Python module containing function used to style plots of the coupling coefficient profiles.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
from matplotlib import gridspec
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# type checking imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib import figure, axes, lines
    import numpy as np


# get tight layout for gridspec, as well as outer labels
def get_tight_layout_outer_labels(
    gs: gridspec.GridSpec,
    fig: 'figure.Figure',
    rect: tuple[float, float, float, float],
) -> None:
    # tight layout for gridspec
    gs.tight_layout(fig, rect=rect)
    # try label outer:
    for ax in fig.axes:
        ax.label_outer()


def get_figure_axes() -> (
    'tuple[figure.Figure, gridspec.GridSpec, axes.Axes, axes.Axes]'
):
    coupling_fig = plt.figure(dpi=300)
    coupling_gs = gridspec.GridSpec(
        2, 1, coupling_fig, right=0.95, wspace=0.0, hspace=0.0
    )
    coupling_ax = coupling_fig.add_subplot(coupling_gs[0, 0])
    coupling_ax_tar_validity = coupling_fig.add_subplot(
        coupling_gs[1, 0], sharex=coupling_ax, sharey=coupling_ax
    )
    coupling_gs.update(wspace=0.0, hspace=0.0)
    return coupling_fig, coupling_gs, coupling_ax, coupling_ax_tar_validity


def get_figure_ax_single() -> (
    'tuple[figure.Figure, gridspec.GridSpec, axes.Axes]'
):
    coupling_fig = plt.figure(dpi=300)
    coupling_gs = gridspec.GridSpec(
        1, 1, coupling_fig, right=0.95, wspace=0.0, hspace=0.0
    )
    coupling_ax = coupling_fig.add_subplot(coupling_gs[0, 0])
    coupling_gs.update(wspace=0.0, hspace=0.0)
    return coupling_fig, coupling_gs, coupling_ax


def adjust_ticks(
    coupling_ax: 'axes.Axes',
    my_brunt_profile: 'np.ndarray | None',
    y_color: str,
    multiple_y: bool,
    label_size: int = 11,
) -> None:
    coupling_ax.xaxis.set_minor_locator(AutoMinorLocator())
    coupling_ax.yaxis.set_minor_locator(AutoMinorLocator())
    if my_brunt_profile is None:
        coupling_ax.tick_params(axis='y', labelsize=label_size)
    else:
        coupling_ax.tick_params(
            axis='y', labelcolor=y_color, labelsize=label_size
        )
    coupling_ax.tick_params(axis='x', labelsize=label_size)
    if multiple_y:
        # make ticks on all sides visible
        coupling_ax.tick_params(top=True, right=True, which='both')
    else:
        # make ticks visible on top
        coupling_ax.tick_params(top=True, which='both')


def set_axis_labels(
    coupling_ax: 'axes.Axes',
    x_axis_label: str,
    y_axis_label: str | None,
    my_brunt_profile: 'np.ndarray | None',
    y_color: str,
    show_x_label: bool,
    label_size: int = 12,
) -> None:
    if show_x_label:
        coupling_ax.set_xlabel(x_axis_label, size=label_size)
    if y_axis_label is not None:
        if my_brunt_profile is None:
            coupling_ax.set_ylabel(y_axis_label, size=label_size)
        else:
            coupling_ax.set_ylabel(y_axis_label, color=y_color, size=label_size)


def add_legend(
    coupling_fig: 'figure.Figure',
    my_plot_data_handles: 'list[lines.Line2D]',
    legend_entries: 'list[str] | None',
    multiple_y: bool,
    plot_legend: bool,
) -> None:
    # add a legend, if requested
    if multiple_y and plot_legend:
        coupling_fig.legend(
            handles=my_plot_data_handles,
            labels=legend_entries,
            loc=5,
            fancybox=True,
            shadow=True,
            fontsize=12,
        )


def x_axis_limits(coupling_ax: 'axes.Axes') -> None:
    coupling_ax.set_xlim(left=0, right=1)


def add_brunt_label(
    brunt_label: bool, coupling_fig: 'figure.Figure', brunt_color: str
) -> None:
    if brunt_label:
        # add sup y-axis label
        coupling_fig.text(
            0.955,
            0.5,
            r'$N^2$ (Hz)',
            color=brunt_color,
            size=12,
            rotation='vertical',
            ha='left',
            va='center',
        )


def brunt_formatting(brunt_ax: 'axes.Axes', brunt_color: str) -> None:
    # set axis scale
    brunt_ax.set_yscale('log')
    # adjust ticks
    brunt_ax.tick_params(axis='y', labelcolor=brunt_color, labelsize=11)
    # adjust y-axis limit
    brunt_ax.set_ylim(bottom=1e-19)


def custom_color_cycle(
    current_index: int, colors_list: list, skip_nr: int = 0
) -> tuple[str, int]:
    """Returns the color name after skipping 'skip_nr' amount of colors.

    Parameters
    ----------
    current_index : int
        The current index of the colors list.
    colors_list : list
        Contains the matplotlib colors used in this property cycle.
    skip_nr : int, optional
        Amount of colors that need to be skipped; by default 0.

    Returns
    -------
    str
        Name of the color that will be used to plot.
    int
        The new color index (needs to be updated!) for the next plot.
    """
    # compute the correct not-out-of-bounds indices
    mod_len = len(colors_list)
    color_idx = (current_index + skip_nr) % mod_len
    next_try_color_idx = (current_index + skip_nr + 1) % mod_len
    # return the color and the next index
    return colors_list[color_idx], next_try_color_idx
