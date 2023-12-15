"""Python module containing generic plot functions, which are necessary to generate the kernel plots.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator

# type checking imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib import figure, gridspec


def add_figure_legend_update_layout_and_save(
    fig: 'figure.Figure',
    ax: 'Axes | list[Axes]',
    gs: 'gridspec.GridSpec',
    leftmost_x_ticks: list[float],
    save_name: str = 'test_kernel.png',
    save_format: str = 'png',
    plot_legend: bool = True,
    use_tight_layout: bool = True,
    multiple_ax: bool = False,
    nr_bins_x: int = 3,
):
    # add legend on top of figure, if necessary
    if plot_legend:
        if TYPE_CHECKING:
            assert isinstance(ax, Axes)
        ax.legend(
            fontsize=14,
            loc='lower left',
            bbox_to_anchor=(0, 1.02, 1, 0.2),
            mode='expand',
            fancybox=True,
            shadow=False,
            ncol=2,
            borderaxespad=0,
        )
    # set tight layout, if necessary
    if use_tight_layout:
        gs.tight_layout(figure=fig)
    # prune values
    if multiple_ax:
        if TYPE_CHECKING:
            assert isinstance(ax, list)
        # set number of labels/bins to 'nr_bins_x'
        nr_of_bins = nr_bins_x
        # adjust locators to prune
        _count = 0
        _most = len(ax)
        for _a in ax:
            if _count == 0:
                _a.xaxis.set_ticks(leftmost_x_ticks)
            elif _count > 0:
                _a.xaxis.set_major_locator(
                    MaxNLocator(nbins=nr_of_bins - 1, prune='lower')
                )
            elif _count == _most - 1:
                _a.xaxis.set_major_locator(
                    MaxNLocator(nbins=nr_of_bins, prune='lower')
                )
            _count += 1
    # save the figure
    fig.savefig(
        save_name,
        dpi='figure',
        format=save_format,
        bbox_inches='tight',
        transparent=True,
    )


def add_individual_figure_legend_and_title(
    sub_axis, my_title, ncol=2, fontsize=14, set_title=True
):
    # add individual legend
    sub_axis.legend(
        fontsize=12,
        loc='best',
        fancybox=False,
        shadow=False,
        ncol=ncol,
        frameon=False,
    )
    # add title, if necessary
    if set_title:
        sub_axis.set_title(my_title, fontdict={'fontsize': fontsize}, pad=15)
