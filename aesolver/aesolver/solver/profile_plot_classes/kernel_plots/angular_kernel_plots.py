"""Python module containing functions used to generate angular kernel plots.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import numpy as np

# intra-package import
from .kernel_figure import kernel_fig
from .generic_functions import (
    add_individual_figure_legend_and_title,
    add_figure_legend_update_layout_and_save,
)

# type checking imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...profile_type_help import AngularKernelDict
    from ...profile_helper_classes import AngularKernelData
    from matplotlib import axes, gridspec, figure
    from typing import TypedDict
    from pathlib import Path

    class AngKernDict(TypedDict):
        fig: figure.Figure
        ax: axes.Axes | list[axes.Axes]
        gs: gridspec.GridSpec


def _angular_kernel_fig(
    ylabel_number: int,
    nr_plot_cols: int = 1,
    nr_plot_rows: int = 1,
    figsize: tuple[float | int, float | int] = (14.0, 12.0),
) -> 'AngKernDict':
    # get the figure, axes and gridspec
    fig, ax, gs = kernel_fig(
        ylabel_number=ylabel_number,
        x_axis_limits=[-1, 1],
        x_label=r'$\mu$',
        y_label_var_part=r'k\,\mu',
        nr_plot_cols=nr_plot_cols,
        nr_plot_rows=nr_plot_rows,
        figsize=figsize,
        cc_contribution_data=None,
        tar_valid=None,
    )
    # return the dictionary containing these objects
    return {'fig': fig, 'ax': ax, 'gs': gs}


def create_angular_contribution_figures_for_specific_contribution(
    ylabel_number: int,
    contribution_dictionary: 'dict[str, AngularKernelDict]',
    contribution_data: 'AngularKernelData',
    nr_plot_cols: list[int],
    angular_kernel_path: 'str | Path',
    model_name_part: str,
    single_entry: bool = False,
) -> None:
    # create list containing the figures, axes and gridspecs for the different angular kernels
    fig_ax_gs_list = [
        _angular_kernel_fig(ylabel_number=ylabel_number, nr_plot_cols=_nr)
        for _nr in nr_plot_cols
    ]
    # get the list of axes for plotting purposes
    axes_list = []
    for _dd in fig_ax_gs_list:
        my_ax = _dd['ax']
        if TYPE_CHECKING:
            assert isinstance(my_ax, list)
        for a in my_ax:
            axes_list.append(a)
    # plot data you want to plot on figure, with correct labeling
    if single_entry:
        _create_single_figure_angular_contribution(
            contribution_dictionary=contribution_dictionary,
            axes_list=axes_list,
            angular_kernel_path=angular_kernel_path,
            model_name_part=model_name_part,
            ylabel_number=ylabel_number,
            fig_ax_gs_list=fig_ax_gs_list,
            contribution_data=contribution_data,
        )
    else:
        _create_multiple_figures_angular_contributions(
            contribution_dictionary=contribution_dictionary,
            axes_list=axes_list,
            angular_kernel_path=angular_kernel_path,
            model_name_part=model_name_part,
            ylabel_number=ylabel_number,
            fig_ax_gs_list=fig_ax_gs_list,
            contribution_data=contribution_data,
        )


def _create_single_figure_angular_contribution(
    contribution_dictionary: 'dict[str, AngularKernelDict]',
    contribution_data: 'AngularKernelData',
    axes_list: 'list[axes.Axes]',
    angular_kernel_path: 'str | Path',
    model_name_part: str,
    ylabel_number: int,
    fig_ax_gs_list: 'list[AngKernDict]',
) -> None:
    for _val in contribution_dictionary.values():
        # construct legend entry, ADJUSTING FOR INCORRECT DOUBLE ESCAPING AND REMOVING UNNECESSARY STRING QUOTES, all the while ensuring we add brackets for non-symmetric terms...
        # ENSURE WE USE THE CORRECT LEGEND ENTRY BASED ON WHETHER A LOGARITHM IS
        if _val['symmetric']:
            legend_entry = (
                rf"[{_val['legend_string']}]".replace(r"'$", r'$')  # type: ignore (non-required key)
                .replace(r"$'", r'$')
                .encode()
                .decode('unicode_escape')
            )
        else:
            legend_entry = (
                rf"{_val['legend_string']}".replace(r"'$", r'$')  # type: ignore (non-required key)
                .replace(r"$'", r'$')
                .encode()
                .decode('unicode_escape')
            )
        # plot the data
        axes_list[0].plot(
            (np.ones_like(_val['kernel']) * contribution_data._mu_vals).T,
            _val['kernel'].T,
            label=legend_entry,
            lw=0.75,
        )
    # add figure legend to plot itself
    add_individual_figure_legend_and_title(
        sub_axis=axes_list[0], my_title='', set_title=False, ncol=1
    )
    # update layout of, and save the figure
    add_figure_legend_update_layout_and_save(
        fig=fig_ax_gs_list[0]['fig'],
        ax=axes_list[0],
        gs=fig_ax_gs_list[0]['gs'],
        save_name=f'{angular_kernel_path}/cc_contribution_{ylabel_number}_radial_kernels{model_name_part}.png',
        save_format='png',
        plot_legend=False,
        leftmost_x_ticks=[-1.0, 0.0, 1.0],
    )


def _create_multiple_figures_angular_contributions(
    contribution_dictionary: 'dict[str, AngularKernelDict]',
    contribution_data: 'AngularKernelData',
    axes_list: 'list[axes.Axes]',
    angular_kernel_path: 'str | Path',
    model_name_part: str,
    ylabel_number: int,
    fig_ax_gs_list: 'list[AngKernDict]',
) -> None:
    for _i, _val in enumerate(contribution_dictionary.values()):
        # create legend according to shape of items
        if len(my_leg := _val['legend_string']) > 1:  # type: ignore (non-required key)
            # create list of legend entries
            legend_entry = (
                [rf'{_l}' for _l in my_leg]
                if _val['symmetric']
                else [rf'[{_l}]' for _l in my_leg]
            )
        else:
            # create single legend entry
            legend_entry = (
                rf'{my_leg[0]}' if _val['symmetric'] else rf'[{my_leg[0]}]'
            )
        # plot the contribution on the appropriate axes
        axes_list[_i].plot(
            (np.ones_like(_val['kernel']) * contribution_data._mu_vals).T,
            _val['kernel'].T,
            label=legend_entry,
            lw=0.75,
        )
        # add individual legend and title to the subplot
        add_individual_figure_legend_and_title(
            sub_axis=axes_list[_i], my_title='', set_title=False, ncol=1
        )
    # update layout of, and save the figures
    for _i, _dd in enumerate(fig_ax_gs_list):
        add_figure_legend_update_layout_and_save(
            fig=_dd['fig'],
            ax=_dd['ax'],
            gs=_dd['gs'],
            save_name=f'{angular_kernel_path}/cc_contribution_{ylabel_number}_angular_kernels_nr_{_i + 1}{model_name_part}.png',
            save_format='png',
            plot_legend=False,
            use_tight_layout=False,
            multiple_ax=True,
            leftmost_x_ticks=[-1.0, 0.0, 1.0],
        )
