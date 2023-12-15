"""Python module containing class that generates the overview plot.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# submodule imports
from .plot import plot_predicted_data_on_target_figure, format_overview_plot

# type checking imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..quadratic_plotter import QuadraticAEPlotter
    from matplotlib import figure, axes
    import numpy.typing as npt


class OverviewFigure:
    """Class containing the necessary methods to generate the overview figure."""

    # attribute type declarations
    _fig: 'figure.Figure | None'
    _ax: 'axes.Axes | None'
    _plot_object: 'QuadraticAEPlotter'
    _mask_stable: 'npt.NDArray[np.bool_]'
    _mask_ae_stable: 'npt.NDArray[np.bool_]'
    _mask_stable_valid: 'npt.NDArray[np.bool_]'
    _negative_value_mask: 'npt.NDArray[np.bool_]'

    def __init__(self, plotter: 'QuadraticAEPlotter') -> None:
        # set initial attributes
        self._set_none_attrs()
        # store the plot object
        self._plot_object = plotter
        # store the masks
        self._unpack_mask_lists_for_plotting()
        # extract data from the data dict lists
        self._extract_data_from_data_dict_list()
        # store save path and save info
        self._store_save_path_and_info()

    # define getters and setters

    def _set_none_attrs(self):
        """Sets the initial attributes to None, wherever possible."""
        # initialize the overview category information to None

        # initialize the overview plot information to None
        self._fig = None
        self._ax = None

    def _store_save_path_and_info(self) -> None:
        # extract the data from the plotter object
        generic_dict = self._plot_object._generic_save_data_dictionary
        specific_dict = self._plot_object._specific_save_data_dictionary
        # create the file name template based on the dictionary information
        file_template_name = f'{specific_dict['overview_save_name']}{f"_{extra_part}" if len(extra_part := generic_dict["extra_specifier"]) > 0 else ""}'
        # get the save directory path
        if len(my_subdir := specific_dict['overview_subdir']) > 0:
            save_dir_path = generic_dict['save_dir'] / my_subdir
            save_dir_path.mkdir(parents=True, exist_ok=True)
        else:
            save_dir_path = generic_dict['save_dir']
        # generate the list of save names
        self._save_paths = [
            Path(f'{save_dir_path}/{file_template_name}.{_suffix}')
            for _suffix in generic_dict['save_formats']
        ]

    # unpack the masks stored in separate lists
    def _unpack_mask_lists_for_plotting(self) -> None:
        # concatenate the mask lists into a single masking array
        self._mask_stable = np.concatenate(self._plot_object._mask_list_stable)
        self._mask_ae_stable = np.concatenate(
            self._plot_object._mask_list_ae_stable
        )
        self._mask_stable_valid = np.concatenate(
            self._plot_object._mask_list_stable_valid
        )

    def _extract_data_from_data_dict_list(self) -> None:
        # create a local reference to the data dict list
        _list_data_dict = self._plot_object._list_data_dict
        # concatenate the negative value mask
        self._negative_value_mask = np.concatenate(
            [x['neg_mask'] for x in _list_data_dict]
        )
        # concatenate the minimal frequency list
        self._min_freq_stack = np.concatenate(
            [x['min_inertial_freq_stack'] for x in _list_data_dict]
        )
        # concatenate the amplitude ratio list
        self._amp_ratio_stack = np.concatenate(
            [x['surf_ratio_daughter_parent_OG_SHAPE'] for x in _list_data_dict]
        )

    # generate plot
    def generate_plot(self, fig_size: tuple[float, float]) -> None:
        """Generate the resonance sharpness plot/figure.

        Parameters
        ----------
        fig_size : tuple[float, float]
            Specifies the matplotlib figure size.
        """
        # generate the figure and axes objects
        self._fig, self._ax = plt.subplots(
            dpi=300, nrows=1, ncols=1, figsize=fig_size
        )
        # dynamically link x-tick positions (only share locator: formatter is set separately as a FuncFormatter)
        sec_ax = self._ax.twiny()
        self._ax._shared_axes['x'].join(self._ax, sec_ax)
        # dynamically minor axis ticks (locator and formatter)
        self._ax.xaxis.minor = sec_ax.xaxis.minor
        # self._ax.get_shared_x_axes().joined(self._ax, my_second_axis)
        # delegate plot creation to function defined in other module
        plot_predicted_data_on_target_figure(figure_object=self)
        # format the generated plot
        format_overview_plot(my_axis=self._ax, my_second_axis=sec_ax)
        # set tight layout
        self._fig.tight_layout()

    # save an individual plot
    def _save_plot(self, save_path: Path, dpi: int) -> None:
        """Save an individual plot with a specific requested dpi.

        Parameters
        ----------
        save_path : Path
            Pathlib Path to individual save file.
        dpi : int
            Dpi of the figure.
        """
        # get the file suffix, so that we know the save format
        save_suffix = save_path.suffix[1:]
        # save the figure, if it exists
        if self._fig is not None:
            self._fig.savefig(
                fname=save_path,
                format=save_suffix,
                dpi=dpi,
                bbox_inches='tight',
                pad_inches=0.0,
            )

    # save plot
    def save_plots(self, dpi: int) -> None:
        """Save the plot(s) with a specific requested dpi.

        Parameters
        ----------
        dpi : int
            Dpi of the figure.
        """
        # loop over the save paths, generating a figure for each of the different file save suffixes
        for _s_p in self._save_paths:
            # save individual plot
            self._save_plot(save_path=_s_p, dpi=dpi)
