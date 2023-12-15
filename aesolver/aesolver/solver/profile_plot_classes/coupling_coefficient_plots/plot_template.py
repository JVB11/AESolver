"""Python module containing function used to generate a template plot of the coupling coefficient profiles.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import numpy as np
import matplotlib.pyplot as plt

# intra-package imports
from .plot_styling import (
    get_tight_layout_outer_labels,
    get_figure_axes,
    get_figure_ax_single,
    adjust_ticks,
    set_axis_labels,
    add_brunt_label,
    add_legend,
    x_axis_limits,
    brunt_formatting,
    custom_color_cycle,
)

# type checking imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cycler import Cycler
    from matplotlib import figure, lines, axes
    import numpy.typing as npt
    from typing import Literal
    from collections.abc import Sequence


def get_contribution_due_to_convective_zone(
    zone_indices: tuple[int, int], quantity: np.ndarray
) -> float:
    """_summary_

    Parameters
    ----------
    zone_indices : _type_
        _description_
    quantity : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    # get start and end values of the quantity profile inside the convective zone
    start_value = quantity[zone_indices[0]]
    end_value = quantity[zone_indices[1]]
    # return the difference = contribution due to convective zone
    return end_value - start_value


# single profile figure template function
def single_integrated_radial_profile_figure_template_function(
    fractional_radius: np.ndarray,
    y_values: np.ndarray | list[np.ndarray],
    y_axis_label: None | str,
    x_axis_label: str = 'r / R',
    my_brunt_profile=None,
    y_color: str = 'xkcd:darkblue',
    brunt_color: str = 'xkcd:red',
    brunt_label: bool = True,
    plot_line_style: str = '-',
    show_x_label: bool = True,
    plot_legend: bool = False,
    legend_entries: None | list[str] = None,
) -> 'figure.Figure':
    # PLOT:
    # define the figure and its axes
    coupling_fig, coupling_gs, coupling_ax = get_figure_ax_single()
    # check for multiple y axis data situations
    multiple_y = isinstance(y_values, list)
    # keep track of multiple data labels, if necessary
    my_plot_data_handles: 'list[lines.Line2D]' = []
    # plot the data
    if multiple_y:
        # get the color cycling list
        prop_cycle: 'Cycler' = plt.rcParams['axes.prop_cycle']
        my_colors = prop_cycle.by_key()['color']
        # plot entries
        if legend_entries is None:
            color_index = 0
            for _i, _y_val in enumerate(y_values):
                # skip SOME COLORS
                if _i == 2:
                    my_color, color_index = custom_color_cycle(
                        current_index=color_index,
                        colors_list=my_colors,
                        skip_nr=2,
                    )
                elif _i == 3:
                    my_color, color_index = custom_color_cycle(
                        current_index=color_index,
                        colors_list=my_colors,
                        skip_nr=4,
                    )
                else:
                    my_color, color_index = custom_color_cycle(
                        current_index=color_index,
                        colors_list=my_colors,
                        skip_nr=0,
                    )
                (mp,) = coupling_ax.plot(
                    fractional_radius,
                    _y_val,
                    plot_line_style,
                    lw=0.75,
                    c=my_color,
                )
                my_plot_data_handles.append(mp)
        else:
            color_index = 0
            for _i, (_y_val, _l_entry) in enumerate(
                zip(y_values, legend_entries)
            ):
                # skip SOME COLORS
                if _i == 2:
                    my_color, color_index = custom_color_cycle(
                        current_index=color_index,
                        colors_list=my_colors,
                        skip_nr=2,
                    )
                elif _i == 3:
                    my_color, color_index = custom_color_cycle(
                        current_index=color_index,
                        colors_list=my_colors,
                        skip_nr=4,
                    )
                else:
                    my_color, color_index = custom_color_cycle(
                        current_index=color_index,
                        colors_list=my_colors,
                        skip_nr=0,
                    )
                (mp,) = coupling_ax.plot(
                    fractional_radius,
                    _y_val,
                    plot_line_style,
                    label=_l_entry,
                    lw=0.75,
                    c=my_color,
                )
                my_plot_data_handles.append(mp)
    else:
        coupling_ax.plot(
            fractional_radius, y_values, plot_line_style, c=y_color, lw=0.75
        )
    # set axis labels
    set_axis_labels(
        coupling_ax=coupling_ax,
        x_axis_label=x_axis_label,
        y_axis_label=y_axis_label,
        my_brunt_profile=my_brunt_profile,
        y_color=y_color,
        show_x_label=show_x_label,
        label_size=14,
    )
    # adjust ticks
    adjust_ticks(
        coupling_ax=coupling_ax,
        my_brunt_profile=my_brunt_profile,
        y_color=y_color,
        multiple_y=multiple_y,
        label_size=12,
    )
    # show the Brunt-Väisälä frequency (squared) profile, if provided and when not displaying multiple profiles at once
    if (my_brunt_profile is not None) and (not multiple_y):
        # get the brunt axis
        brunt_ax = coupling_ax.twinx()
        # plot the brunt profile
        brunt_ax.plot(  # type: ignore
            fractional_radius, my_brunt_profile, c=brunt_color, lw=0.75
        )
        # set axis label
        add_brunt_label(
            brunt_label=brunt_label,
            coupling_fig=coupling_fig,
            brunt_color=brunt_color,
        )
        # set axis scale, adjust ticks and adjust y-axis limit
        brunt_formatting(brunt_ax=brunt_ax, brunt_color=brunt_color)  # type: ignore
    # add a legend, if requested
    add_legend(
        coupling_fig=coupling_fig,
        my_plot_data_handles=my_plot_data_handles,
        legend_entries=legend_entries,
        multiple_y=multiple_y,
        plot_legend=plot_legend,
    )
    # set x-axis limits
    x_axis_limits(coupling_ax=coupling_ax)
    # tight layout + label outer
    get_tight_layout_outer_labels(
        gs=coupling_gs, fig=coupling_fig, rect=(0, 0, 0.95, 1)
    )
    # return the figure for saving
    return coupling_fig


# double masking figure template function
def double_masking_integrated_radial_profile_figure_template_function(
    radial_coordinate: np.ndarray,
    y_ax_data: 'npt.NDArray[np.float64] | list[npt.NDArray[np.float64]]',
    y_ax_label: list[str],
    convection_zone_selection_indices: 'list[tuple[int, int]] | list[tuple[Literal[0], int]]',
    fully_invalid: bool,
    convection_zone_masks: 'Sequence[slice | None]',
    validity_zone_masks: 'Sequence[slice | None]',
    my_brunt: None | np.ndarray = None,
    tar_validity_mask: 'npt.NDArray[np.bool_] | slice' = np.s_[:],
    use_legend: bool = False,
    y_legend_entries: None | list[str] = None,
) -> 'tuple[float, float, figure.Figure, list[float]] | tuple[list[float], list[float], figure.Figure, list[list[float]]]':
    # ACTIONS FOR MULTIPLE Y AXIS DATA:
    if isinstance(y_ax_data, list):
        # -- compute contributions due TAR-invalid regions
        # get the contributions to the (integrated) y-axis data due to the TAR-invalid zones, assuming we have a model with a convective core!
        contributions = [
            [
                get_contribution_due_to_convective_zone(
                    zone_indices=x, quantity=_y_ax_d
                )
                for x in convection_zone_selection_indices
            ]
            for _y_ax_d in y_ax_data
        ]
        # get the summed contributions
        _summed_contributions = [np.cumsum(_cont) for _cont in contributions]
        # get the values of the (integrated) y-axis data in the TAR-invalid zones (i.e. the indices of the values right before those zones)
        _y_invalid = [
            [y_ax[x[0]] for x in convection_zone_selection_indices]
            for y_ax in y_ax_data
        ]
        # compute a y-profile that masks the core
        _adjusted_core_profile = [np.zeros_like(y_ax) for y_ax in y_ax_data]
        if not fully_invalid:
            _core_contribution = [x[0] for x in _summed_contributions]
            _core_mask = [
                np.s_[convection_zone_selection_indices[0][1] + 1 :]
                for _ in y_ax_data
            ]
            for _c_mask, _y_ax, _core_contrib, _adj_core in zip(
                _core_mask,
                y_ax_data,
                _core_contribution,
                _adjusted_core_profile,
            ):
                _adj_core[_c_mask] = _y_ax[_c_mask] - _core_contrib
        # compute a y-profile that masks all TAR-invalid zones (including the core!)
        _adjusted_tar_profile = [np.zeros_like(y_ax) for y_ax in y_ax_data]
        if not fully_invalid:
            for _adj_tar, _y_inv, _summed_contrib in zip(
                _adjusted_tar_profile, _y_invalid, _summed_contributions
            ):
                for _i, (_convection_mask, _conv) in enumerate(
                    zip(convection_zone_masks, _y_inv)
                ):
                    if _i > 0:
                        _adj_tar[_convection_mask] = (
                            _conv - _summed_contrib[_i - 1]
                        )
                    else:
                        _adj_tar[_convection_mask] = _conv
            for _y_ax, _adj_tar, _summed_contrib in zip(
                y_ax_data, _adjusted_tar_profile, _summed_contributions
            ):
                for _adjustment_value, _valid_mask in zip(
                    _summed_contrib, validity_zone_masks
                ):
                    _adj_tar[_valid_mask] = (
                        _y_ax[_valid_mask] - _adjustment_value
                    )

        # PLOT:
        # define the figure and its axes
        (
            coupling_fig,
            coupling_gs,
            coupling_ax,
            coupling_ax_tar_validity,
        ) = get_figure_axes()

        # 1) regular plot without any masking
        create_individual_figure_coupling_profile(
            coupling_fig,
            coupling_ax,
            fractional_radius=radial_coordinate,
            y_values=y_ax_data,
            y_axis_label=y_ax_label[0],
            tar_validity_mask=tar_validity_mask,
            my_brunt_profile=None,
            plot_legend=False,
            legend_entries=y_legend_entries,
        )
        # 2) plot where all zones that do not satisfy the TAR are masked
        create_individual_figure_coupling_profile(
            coupling_fig,
            coupling_ax_tar_validity,
            fractional_radius=radial_coordinate,
            y_values=_adjusted_tar_profile,
            y_axis_label=y_ax_label[1],
            tar_validity_mask=tar_validity_mask,
            my_brunt_profile=my_brunt,
            plot_line_style='-',
            brunt_label=True,
            show_x_label=True,
            plot_legend=use_legend,
            legend_entries=y_legend_entries,
        )

        # tight layout for gridspec to make room for legend
        get_tight_layout_outer_labels(
            gs=coupling_gs, fig=coupling_fig, rect=(0.0, 0.0, 0.825, 1.0)
        )

        # return the adjusted integrated y-term values
        return (
            [y_ax[-1] for y_ax in y_ax_data],
            [
                y_ax[-1] - _summed_contrib[-1]
                for y_ax, _summed_contrib in zip(
                    y_ax_data, _summed_contributions
                )
            ],
            coupling_fig,
            contributions,
        )
    # ACTIONS FOR SINGLE Y AXIS DATA:
    else:
        # -- compute contributions due TAR-invalid regions
        # get the contributions to the (integrated) y-axis data due to the TAR-invalid zones, assuming we have a model with a convective core!
        contributions = [
            get_contribution_due_to_convective_zone(
                zone_indices=x, quantity=y_ax_data
            )
            for x in convection_zone_selection_indices
        ]
        # get the summed contributions
        _summed_contributions = np.cumsum(contributions)
        # get the values of the (integrated) y-axis data in the TAR-invalid zones (i.e. the indices of the values right before those zones)
        _y_invalid = [
            y_ax_data[x[0]] for x in convection_zone_selection_indices
        ]
        # compute a y-profile that masks the core
        _adjusted_core_profile = np.zeros_like(y_ax_data)
        if not fully_invalid:
            _core_contribution = _summed_contributions[0]
            _core_mask = np.s_[convection_zone_selection_indices[0][1] + 1 :]
            _adjusted_core_profile[_core_mask] = (
                y_ax_data[_core_mask] - _core_contribution
            )
        # compute a y-profile that masks all TAR-invalid zones (including the core!)
        _adjusted_tar_profile = np.zeros_like(y_ax_data)
        if not fully_invalid:
            for _i, (_convection_mask, _conv) in enumerate(
                zip(convection_zone_masks, _y_invalid)
            ):
                if _i > 0:
                    _adjusted_tar_profile[_convection_mask] = (
                        _conv - _summed_contributions[_i - 1]
                    )
                else:
                    _adjusted_tar_profile[_convection_mask] = _conv
            for _adjustment_value, _valid_mask in zip(
                _summed_contributions, validity_zone_masks
            ):
                _adjusted_tar_profile[_valid_mask] = (
                    y_ax_data[_valid_mask] - _adjustment_value
                )

        # PLOT:
        # define the figure and its axes
        (
            coupling_fig,
            coupling_gs,
            coupling_ax,
            coupling_ax_tar_validity,
        ) = get_figure_axes()

        # 1) regular plot without any masking
        create_individual_figure_coupling_profile(
            coupling_fig,
            coupling_ax,
            fractional_radius=radial_coordinate,
            y_values=y_ax_data,
            y_axis_label=y_ax_label[0],
            tar_validity_mask=tar_validity_mask,
            my_brunt_profile=my_brunt,
        )
        # 2) plot where all zones that do not satisfy the TAR are masked
        create_individual_figure_coupling_profile(
            coupling_fig,
            coupling_ax_tar_validity,
            fractional_radius=radial_coordinate,
            y_values=_adjusted_tar_profile,
            y_axis_label=y_ax_label[1],
            tar_validity_mask=tar_validity_mask,
            my_brunt_profile=my_brunt,
            plot_line_style='-',
            brunt_label=True,
            show_x_label=True,
        )

        # tight layout + label outer
        get_tight_layout_outer_labels(
            gs=coupling_gs, fig=coupling_fig, rect=(0, 0, 0.95, 1)
        )

        # return the adjusted integrated y-term value
        return (
            y_ax_data[-1],
            y_ax_data[-1] - _summed_contributions[-1],
            coupling_fig,
            contributions,
        )


# plot individual figure
def create_individual_figure_coupling_profile(
    coupling_fig: 'figure.Figure',
    coupling_ax: 'axes.Axes',
    fractional_radius: np.ndarray,
    y_values: np.ndarray | list[np.ndarray],
    y_axis_label: None | str,
    x_axis_label: str = 'r / R',
    tar_validity_mask=None,
    my_brunt_profile=None,
    y_color: str = 'xkcd:darkblue',
    brunt_color: str = 'xkcd:red',
    brunt_label: bool = False,
    plot_line_style: str = '-',
    show_x_label: bool = False,
    plot_shaded_domain: bool = True,
    plot_legend: bool = False,
    legend_entries: None | list[str] = None,
):
    # check for multiple y axis data situations
    multiple_y = isinstance(y_values, list)
    # keep track of multiple data labels, if necessary
    my_plot_data_handles = []
    # plot the data
    if multiple_y:
        # plot entries
        if legend_entries is None:
            for _y_val in y_values:
                (mp,) = coupling_ax.plot(
                    fractional_radius, _y_val, plot_line_style, lw=0.75
                )
                my_plot_data_handles.append(mp)
        else:
            for _y_val, _l_entry in zip(y_values, legend_entries):
                (mp,) = coupling_ax.plot(
                    fractional_radius,
                    _y_val,
                    plot_line_style,
                    label=_l_entry,
                    lw=0.75,
                )
                my_plot_data_handles.append(mp)
    else:
        coupling_ax.plot(
            fractional_radius, y_values, plot_line_style, c=y_color, lw=0.75
        )
    # add shaded domain showing where TAR is valid
    if plot_shaded_domain:
        coupling_ax.fill_between(
            fractional_radius,
            0,
            1,
            where=tar_validity_mask,
            transform=coupling_ax.get_xaxis_transform(),
            color='k',
            alpha=0.05 if multiple_y else 0.1,
            lw=0.0,
        )
    # set axis labels
    set_axis_labels(
        coupling_ax=coupling_ax,
        x_axis_label=x_axis_label,
        y_axis_label=y_axis_label,
        my_brunt_profile=my_brunt_profile,
        y_color=y_color,
        show_x_label=show_x_label,
    )
    # adjust ticks
    adjust_ticks(
        coupling_ax=coupling_ax,
        my_brunt_profile=my_brunt_profile,
        y_color=y_color,
        multiple_y=multiple_y,
    )
    # show the Brunt-Väisälä frequency (squared) profile, if provided and when not displaying multiple profiles at once
    if (my_brunt_profile is not None) and (not multiple_y):
        # get the brunt axis
        brunt_ax = coupling_ax.twinx()
        # plot the brunt profile
        brunt_ax.plot(  # type: ignore
            fractional_radius, my_brunt_profile, c=brunt_color, lw=0.75
        )
        # set axis label
        add_brunt_label(
            brunt_label=brunt_label,
            coupling_fig=coupling_fig,
            brunt_color=brunt_color,
        )
        # set axis scale, adjust ticks and adjust y-axis limit
        brunt_formatting(brunt_ax=brunt_ax, brunt_color=brunt_color)  # type: ignore
    # add a legend, if requested
    add_legend(
        coupling_fig=coupling_fig,
        my_plot_data_handles=my_plot_data_handles,
        legend_entries=legend_entries,
        multiple_y=multiple_y,
        plot_legend=plot_legend,
    )
    # set x-axis limits
    x_axis_limits(coupling_ax=coupling_ax)
