"""Module containing the figure class for the resonance sharpness figure.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# submodule imports
from .histogram import get_sharpness_histogram_plot_data
from .plot import generate_sharpness_histogram_plot

# type checking imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib import figure, axes
    import numpy.typing as npt


class ResonanceSharpnessFigure:
    """Class containing the necessary methods to generate the resonance sharpness figure.

    Parameters
    ----------
    TODO
    """

    # attribute type declarations
    _sharpness_all: 'npt.NDArray[np.float64]'
    _sharpness_isolated: 'npt.NDArray[np.float64] | None'
    _sharpness_not_isolated: 'npt.NDArray[np.float64] | None'
    _edges: 'npt.NDArray[np.float64] | None'
    _bin_width: 'np.float64 | None'
    _prob_isolated_log: 'npt.NDArray[np.float64] | None'
    _prob_not_isolated_log: 'npt.NDArray[np.float64] | None'
    _fig: 'figure.Figure | None'
    _ax: 'axes.Axes | None'
    _x_coords: 'npt.NDArray[np.float64] | None'
    _save_paths: 'list[Path]'
    _count_number_dict: dict[str, list[int]]

    def __init__(
        self,
        sharpness_data_dict: 'dict[str, npt.NDArray[np.float64]]',
        count_nr_dict: 'dict[str, list[int]]',
        generic_save_dict: dict[str, str],
        specific_save_dict: dict[str, str],
    ) -> None:
        # set initial attributes
        self._set_none_attrs()
        # store save path and save info
        self._store_save_path_and_info(
            generic_dict=generic_save_dict, specific_dict=specific_save_dict
        )
        # store count number dictionary
        self._count_number_dict = count_nr_dict
        # extract resonance sharpness data
        self._extract_resonance_sharpness(data_dict=sharpness_data_dict)
        # retrieve the data for plotting
        self._get_data_for_plotting()

    # define getters and setters

    @property
    def sharpness_all(self) -> 'npt.NDArray[np.float64]':
        return self._sharpness_all

    @sharpness_all.setter
    def sharpness_all(self, my_val: 'npt.NDArray[np.float64]') -> None:
        self._sharpness_all = my_val

    @property
    def sharpness_isolated(self) -> 'npt.NDArray[np.float64] | None':
        return self._sharpness_isolated

    @sharpness_isolated.setter
    def sharpness_isolated(
        self, my_val: 'npt.NDArray[np.float64] | None'
    ) -> None:
        self._sharpness_isolated = my_val

    @property
    def sharpness_not_isolated(self) -> 'npt.NDArray[np.float64] | None':
        return self._sharpness_not_isolated

    @sharpness_not_isolated.setter
    def sharpness_not_isolated(
        self, my_val: 'npt.NDArray[np.float64] | None'
    ) -> None:
        self._sharpness_not_isolated = my_val

    @property
    def bin_width(self) -> 'np.float64 | None':
        return self._bin_width

    @bin_width.setter
    def bin_width(self, my_val: 'np.float64 | None') -> None:
        self._bin_width = my_val

    @property
    def edges(self) -> 'npt.NDArray[np.float64] | None':
        return self._edges

    @edges.setter
    def edges(self, my_val: 'npt.NDArray[np.float64] | None') -> None:
        self._edges = my_val

    @property
    def prob_isolated_log(self) -> 'npt.NDArray[np.float64] | None':
        return self._prob_isolated_log

    @prob_isolated_log.setter
    def prob_isolated_log(
        self, my_val: 'npt.NDArray[np.float64] | None'
    ) -> None:
        self._prob_isolated_log = my_val

    @property
    def prob_not_isolated_log(self) -> 'npt.NDArray[np.float64] | None':
        return self._prob_not_isolated_log

    @prob_not_isolated_log.setter
    def prob_not_isolated_log(
        self, my_val: 'npt.NDArray[np.float64] | None'
    ) -> None:
        self._prob_not_isolated_log = my_val

    def _set_none_attrs(self):
        """Sets the initial attributes to None, wherever possible."""
        # initialize the resonance sharpness category information to None
        self._sharpness_isolated = None
        self._sharpness_not_isolated = None
        # initialize the resonance sharpness plot information to None
        self._edges = None
        self._bin_width = None
        self._prob_isolated_log = None
        self._prob_not_isolated_log = None
        self._x_coords = None
        self._fig = None
        self._ax = None

    def _store_save_path_and_info(
        self, generic_dict: dict[str, str], specific_dict: dict[str, str]
    ) -> None:
        # create the file name template based on the dictionary information
        file_template_name = f'{specific_dict['sharpness_save_name']}{f"_{extra_part}" if len(extra_part := generic_dict["extra_specifier"]) > 0 else ""}'
        # get the save directory path
        if len(my_subdir := specific_dict['sharpness_subdir']) > 0:
            save_dir_path = generic_dict['save_dir'] / my_subdir
            save_dir_path.mkdir(parents=True, exist_ok=True)
        else:
            save_dir_path = generic_dict['save_dir']
        # generate the list of save names
        self._save_paths = [
            Path(f'{save_dir_path}/{file_template_name}.{_suffix}')
            for _suffix in generic_dict['save_formats']
        ]

    # get resonance sharpness data
    def _extract_resonance_sharpness(
        self, data_dict: 'dict[str, npt.NDArray[np.float64]]'
    ) -> None:
        """Retrieves the information on the resonance sharpness."""
        # store the resonance sharpness information
        self._sharpness_all = data_dict['all']
        self._sharpness_isolated = data_dict['isolated']
        self._sharpness_not_isolated = data_dict['not isolated']

    # get data for plotting
    def _get_data_for_plotting(self) -> None:
        """Retrieves the data used for plotting the resonance sharpness figure/plot."""
        # delegate to function defined in other module
        get_sharpness_histogram_plot_data(resonance_sharpness_object=self)

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
        # delegate plot creation to function defined in other module
        generate_sharpness_histogram_plot(resonance_figure_object=self)

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

    # save plot(s)
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
