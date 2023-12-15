"""Python module containing functions used to generate radial kernel plots.

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
    from ...profile_helper_classes import RadialKernelData
    from ...profile_type_help import RadialKernelDict
    from typing import Sequence
    from matplotlib import axes
    from pathlib import Path
    import numpy.typing as npt


def _radial_kernel_fig(
    cc_contribution_data: 'RadialKernelData',
    ylabel_number: int,
    tar_valid: 'Sequence[bool] | None | npt.NDArray[np.bool_]',
    nr_plot_cols: int = 1,
    nr_plot_rows: int = 1,
    figsize: tuple[int | float, int | float] = (14.0, 12.0),
) -> dict:
    # get the figure, axes and gridspec
    fig, ax, gs = kernel_fig(
        ylabel_number=ylabel_number,
        x_axis_limits=[0, 1],
        x_label=r'r / R',
        y_label_var_part=r'k',
        nr_plot_cols=nr_plot_cols,
        nr_plot_rows=nr_plot_rows,
        figsize=figsize,
        cc_contribution_data=cc_contribution_data,
        tar_valid=tar_valid,
    )
    # return the dictionary containing these objects
    return {'fig': fig, 'ax': ax, 'gs': gs}


def create_radial_contribution_figures_for_specific_contribution(
    ylabel_number: int,
    contribution_dictionary: 'dict[str, RadialKernelDict]',
    contribution_data: 'RadialKernelData',
    nr_plot_cols: list[int],
    radial_kernel_path: 'str | Path',
    model_name_part: str,
    tar_valid: 'Sequence[bool] | None | npt.NDArray[np.bool_]',
    energy_scale_string: str,
    single_entry: bool = False,
) -> None:
    # create list containing the figures, axes and gridspecs for the different radial kernels
    fig_ax_gs_list = [
        _radial_kernel_fig(
            ylabel_number=ylabel_number,
            nr_plot_cols=_nr,
            cc_contribution_data=contribution_data,
            tar_valid=tar_valid,
        )
        for _nr in nr_plot_cols
    ]
    # get the list of axes for plotting purposes
    axes_list = [a for _dd in fig_ax_gs_list for a in _dd['ax']]
    # plot data you want to plot on figure, with correct labeling
    if single_entry:
        _create_single_figure_radial_contribution(
            contribution_data=contribution_data,
            contribution_dictionary=contribution_dictionary,
            axes_list=axes_list,
            radial_kernel_path=radial_kernel_path,
            model_name_part=model_name_part,
            ylabel_number=ylabel_number,
            fig_ax_gs_list=fig_ax_gs_list,
            energy_scale_string=energy_scale_string,
        )
    else:
        _create_multiple_figures_radial_contributions(
            contribution_data=contribution_data,
            contribution_dictionary=contribution_dictionary,
            axes_list=axes_list,
            radial_kernel_path=radial_kernel_path,
            model_name_part=model_name_part,
            ylabel_number=ylabel_number,
            fig_ax_gs_list=fig_ax_gs_list,
            energy_scale_string=energy_scale_string,
        )


def _create_single_figure_radial_contribution(
    contribution_dictionary: 'dict[str, RadialKernelDict]',
    contribution_data: 'RadialKernelData',
    axes_list: 'list[axes.Axes]',
    radial_kernel_path: 'str | Path',
    model_name_part: str,
    ylabel_number: int,
    fig_ax_gs_list: list[dict],
    energy_scale_string: str,
) -> None:
    # plot data you want to plot on figure, with correct labeling
    for _val in contribution_dictionary.values():
        # construct legend entry
        legend_entry = rf"{_val['legend_title']}$\,\,$[{_val['legend'][0]}]{energy_scale_string}"  # type: ignore (not-required key)
        # plot the data
        axes_list[0].plot(
            (np.ones_like(_val['kernel']) * contribution_data.x).T,
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
        save_name=f'{radial_kernel_path}/cc_contribution_{ylabel_number}_radial_kernels{model_name_part}.png',
        save_format='png',
        plot_legend=False,
        leftmost_x_ticks=[0.0, 0.5, 1.0],
    )


def _create_multiple_figures_radial_contributions(
    contribution_dictionary: 'dict[str, RadialKernelDict]',
    contribution_data: 'RadialKernelData',
    axes_list: 'list[axes.Axes]',
    radial_kernel_path: 'str | Path',
    model_name_part: str,
    ylabel_number: int,
    fig_ax_gs_list: list[dict],
    energy_scale_string: str,
) -> None:
    # plot data you want to plot on figure, with correct labeling
    for _i, _val in enumerate(contribution_dictionary.values()):
        # create legend according to shape of items
        if len(my_leg := _val['legend']) > 1:
            # create list of legend entries
            legend_entry = [rf'[{_l}]' for _l in my_leg]
        else:
            # create single legend entry
            legend_entry = rf'[{my_leg[0]}]'
        # plot the contribution on the appropriate axes
        axes_list[_i].plot(
            (np.ones_like(_val['kernel']) * contribution_data.x).T,
            _val['kernel'].T,
            label=legend_entry,
            lw=0.75,
        )
        # add individual legend and title to the subplot
        add_individual_figure_legend_and_title(
            sub_axis=axes_list[_i],
            my_title=rf"{_val['legend_title']}{energy_scale_string}",  # type: ignore (not-required key)
            ncol=1,
        )
    # update layout of, and save the figures
    for _i, _dd in enumerate(fig_ax_gs_list):
        add_figure_legend_update_layout_and_save(
            fig=_dd['fig'],
            ax=_dd['ax'],
            gs=_dd['gs'],
            save_name=f'{radial_kernel_path}/cc_contribution_{ylabel_number}_radial_kernels_nr_{_i + 1}{model_name_part}.png',
            save_format='png',
            plot_legend=False,
            use_tight_layout=False,
            multiple_ax=True,
            leftmost_x_ticks=[0.0, 0.5, 1.0],
        )
