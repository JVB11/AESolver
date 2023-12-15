"""Module containing plotting functions used to generate the resonance sharpness plot.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import logging
from matplotlib.ticker import AutoMinorLocator


logger = logging.getLogger(__name__)


def _get_x_coordinates(resonance_figure_object):
    my_edges = resonance_figure_object.edges
    if my_edges is None:  # GUARD
        resonance_figure_object._x_coords = None
    else:
        resonance_figure_object._x_coords = (
            resonance_figure_object.edges[1:]
            + resonance_figure_object.edges[:-1]
        ) / 2.0


def _plot_sharpness_histograms(
    resonance_figure_object,
    my_axes,
    plot_prob_isolated,
    plot_prob_not_isolated,
    count_nr_dict,
):
    # get references to important plotting variables
    my_x_coords = resonance_figure_object._x_coords
    my_bin_width = resonance_figure_object.bin_width
    # guard the plotting commands
    if my_x_coords is None:
        logger.debug('X coordinates for figure are None.')
    elif my_bin_width is None:
        logger.debug('Bin width for figure is None')
    else:
        if plot_prob_isolated is None:
            logger.debug('Isolated triad bin data is None.')
        else:
            # plot resonance sharpness histogram for isolated triads (GUARDED)
            my_axes.bar(
                x=resonance_figure_object._x_coords,
                height=plot_prob_isolated,
                color='blue',
                width=resonance_figure_object.bin_width,
                alpha=0.5,
                hatch='.',
                label=f'Isolated ({sum(count_nr_dict['isolated'])} triads)',
            )
        if plot_prob_not_isolated is None:
            logger.debug('Non-isolated (stable) triad bin is None.')
        else:
            # plot resonance sharpness histogram for non-isolated triads (GUARDED)
            my_axes.bar(
                x=resonance_figure_object._x_coords,
                height=plot_prob_not_isolated,
                color='#fee090',
                width=resonance_figure_object.bin_width,
                alpha=0.7,
                label=f'Stable & AE-valid ({sum(count_nr_dict['not isolated'])} triads)',
            )


def _add_plot_labels(my_axes):
    my_axes.set_xlabel(
        r'$\left|\delta\omega\,/\,(\gamma_2 + \gamma_3)\right|$', fontsize=14
    )
    my_axes.set_ylabel(r'Empirical Probability (%)', fontsize=14)


def _tick_styling(my_axes):
    # add minor locator
    my_axes.yaxis.set_minor_locator(AutoMinorLocator())
    # style the ticks
    my_axes.tick_params(
        axis='both',
        which='both',
        direction='out',
        bottom=True,
        top=True,
        left=True,
        right=True,
        labelbottom=True,
        labeltop=False,
        labelleft=True,
        labelright=False,
    )


def _add_legend(my_axes):
    my_axes.legend(fontsize=14, frameon=False)


def generate_sharpness_histogram_plot(resonance_figure_object):
    # compute x-coordinates
    _get_x_coordinates(resonance_figure_object=resonance_figure_object)
    # compute probabilities in % (GUARDED)
    isolated_prob = (
        None
        if (my_log := resonance_figure_object.prob_isolated_log) is None
        else my_log * 100.0
    )
    non_isolated_prob = (
        None
        if (my_log := resonance_figure_object.prob_not_isolated_log) is None
        else my_log * 100.0
    )
    # create reference to axes object
    my_ax_all = resonance_figure_object._ax
    # get x-axis log-scale
    my_ax_all.set_xscale('log')
    # plot the histograms (as bar plots)
    _plot_sharpness_histograms(
        my_axes=my_ax_all,
        resonance_figure_object=resonance_figure_object,
        plot_prob_isolated=isolated_prob,
        plot_prob_not_isolated=non_isolated_prob,
        count_nr_dict=resonance_figure_object._count_number_dict,
    )
    # add plot labels
    _add_plot_labels(my_axes=my_ax_all)
    # style ticks
    _tick_styling(my_axes=my_ax_all)
    # add legend
    _add_legend(my_axes=my_ax_all)
    # set tight layout
    resonance_figure_object._fig.tight_layout()
