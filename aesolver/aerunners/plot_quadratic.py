"""Python run script used to compute the quadratic coupling coefficients and show the (stationary) solutions to the quadratic amplitude equations.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
# import os
import sys
import matplotlib.pyplot as plt
from pathlib import Path

# import the plotter class
from aeplotter import QuadraticAEPlotter

# import from custom modules
from generic_parser import GenericParser
from path_resolver import resolve_path_to_file
from log_formatter import set_up_root_logger, adjust_root_logger_level

# type checking imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # import typed dicts
    from .data_modules import SaveInfo, ModeInfo, PlotInfo, LoadInfo
    # import logger class
    from logging import Logger


# set up root logger
logger = set_up_root_logger()


# subclass the GenericParser superclass for use in plotting
# the outcomes of solving quadratic amplitude equations
class QuadraticPlotParser(GenericParser):
    """Subclass of the GenericParser superclass used to parse arguments for generating plots of solve output of the quadratic amplitude equations.

    Parameters
    ----------
    full_inlist_path : Path
        The path to the toml inlist that will be read.
    """

    def __init__(
        self,
        full_inlist_path=None,
    ) -> None:
        super().__init__(
            base_directory_name='amplitude_equations_solver',
            full_inlist_path=full_inlist_path,
            base_dir=None,
            inlist_name=None,
            inlist_dir=None,
            inlist_suffix=None,
        )

    # overload the read arguments method for toml inlists
    def read_toml_args(self) -> 'tuple[SaveInfo, ModeInfo, PlotInfo, LoadInfo]':
        """Read the input arguments from the toml-style inlist.

        Returns
        -------
        my_mode_info_dict : dict
            The dictionary containing information on the modes.
        my_mode_plotting_dict : dict
            The dictionary containing information on the plotting actions.
        """
        # generate the dictionary that will hold the save arguments for the plots/figures
        my_plot_save_dict = self.read_in_to_dict(
            [
                ('save_base_dir', ['saving', 'plot', 'base_dir'], False),
                ('save_dir', ['saving', 'plot', 'save_dir'], False),
                (
                    'extra_specifier_save',
                    ['saving', 'plot', 'extra_specifier'],
                    False,
                ),
                ('save_formats', ['saving', 'plot', 'save_formats'], False),
                ('sharpness_save_name', ['saving', 'plot', 'sharpness', 'save_name'], False),
                ('sharpness_subdir', ['saving', 'plot', 'sharpness', 'save_subdir'], False),
                ('overview_save_name', ['saving', 'plot', 'overview', 'save_name'], False),
                ('overview_subdir', ['saving', 'plot', 'overview', 'save_subdir'], False),
            ]
        )
        # generate the dictionary that will hold the arguments for the solver object
        my_mode_info_dict = self.read_in_to_dict(
            [
                ('rot_pct', ['rotation', 'rot_percent'], False),
                ('base_dir', ['GYRE_selection', 'gyre_dir_path'], False),
                (
                    'alternative_directory_path',
                    ['GYRE_selection', 'alternative', 'directory_path'],
                    False,
                ),
                (
                    'alternative_file_name',
                    ['GYRE_selection', 'alternative', 'file_name'],
                    False,
                ),
                ('verbose', ['verbosity-debug', 'verbose'], False),
                ('debug', ['verbosity-debug', 'debug'], False),
            ]
        )
        # generate the dictionary that holds mode plotting details
        my_mode_plotting_dict = self.read_in_to_dict(
            [
                ('plot_overview', ['plotting', 'plot', 'overview'], False),
                ('verbose', ['verbosity-debug', 'verbose'], False),
                (
                    'plot_coefficient_profile',
                    ['plotting', 'plot', 'coefficient_profile'],
                    False,
                ),
                ('plot_resonance_sharpness', ['plotting', 'plot', 'resonance_sharpness'], False),
                ('fig_size', ['plotting', 'figure', 'size'], False),
                ('fig_dpi', ['plotting', 'figure', 'dpi'], False),
                (
                    'ax_label_size',
                    ['plotting', 'labels', 'ax_label', 'size'],
                    False,
                ),
                (
                    'tick_label_size',
                    ['plotting', 'labels', 'tick_label', 'size'],
                    False,
                ),
                ('tick_size', ['plotting', 'ticks', 'size'], False),
                ('rad_nr', ['plotting', 'misc', 'rad_nr'], False),
                ('show_plots', ['plotting', 'plot', 'show'], False),
                ('cmap', ['plotting', 'colors', 'matplotlib_color_map'], False),
                (
                    'tick_label_pad',
                    ['plotting', 'labels', 'tick_label', 'pad'],
                    False,
                ),
            ]
        )
        # generate a dictionary that holds information for loading files
        my_loading_dict = self.read_in_to_dict(
            [
                (
                    'base_file_names',
                    ['plotting', 'load', 'base_file_names'],
                    False,
                ),
                (
                    'specification_names',
                    ['plotting', 'load', 'specification_names'],
                    False,
                ),
                ('masses', ['plotting', 'load', 'masses'], False),
                ('Xcs', ['plotting', 'load', 'Xcs'], False),
                ('triple_pairs', ['plotting', 'load', 'triple_pairs'], False),
                (
                    'meridional_degrees',
                    ['plotting', 'load', 'meridional_degrees'],
                    False,
                ),
                (
                    'azimuthal_orders',
                    ['plotting', 'load', 'azimuthal_orders'],
                    False,
                ),
                ('file_suffix', ['plotting', 'load', 'file_suffix'], False),
                ('nr_files', ['plotting', 'load', 'nr_files'], False),
                ('output_dir', ['plotting', 'load', 'output_dir'], False),
                ('base_dir', ['plotting', 'load', 'base_dir'], False),
                (
                    'use_standard_format',
                    ['plotting', 'load', 'use_custom_name'],
                    True,
                ),
            ]
        )
        # return the information dictionaries
        return (
            my_plot_save_dict,
            my_mode_info_dict,
            my_mode_plotting_dict,
            my_loading_dict,
        )

def get_path_to_inlist(
    sys_arguments: list,
    inlist_file_name: str = 'inlist',
    inlist_file_suffix: str = 'toml',
) -> Path:
    # generate relevant properties used to create the path to the inlist file
    match len(sys_arguments):
        case 4:
            # construct file name from command line input arguments
            full_file_name = f'{sys_arguments[1]}.{sys_arguments[2]}'
            # store the relative path to the inlist directory from this file
            relative_inlist_path = f'{sys_arguments[3]}'
        case 1:
            # construct full file name from default arguments to this function (i.e. use the default inlist)
            full_file_name = f'{inlist_file_name}.{inlist_file_suffix}'
            # store the relative path to the inlist directory from this file
            relative_inlist_path = './inlists/'
        case _:
            raise RuntimeError(f'A number of input arguments ({len(sys_arguments)}) was passed to this script for which the behavior is undefined.\n\nPlease either specify:\n*) inlist file name, inlist file suffix and relative path to the inlist file from the repository base directory (to read a custom inlist)\n*) no command line arguments (to use the default inlist).\n\nOther options have not yet been implemented.')
    # now resolve the path to the inlist and return it
    return resolve_path_to_file(
        file_name=full_file_name,
        default_path=relative_inlist_path,
        sys_arguments=sys_arguments,
    )
    
    
def _get_and_parse_elements(my_logger: 'Logger') -> 'tuple[SaveInfo | None, PlotInfo, LoadInfo]':
    # check if any command line arguments are passed, and retrieve inlist name and directory
    inlist_path = get_path_to_inlist(sys_arguments=sys.argv)
    # initialize the parser object
    _my_solve_parser = QuadraticPlotParser(
        full_inlist_path=inlist_path
    )
    # read the input arguments
    (
        save_info,
        mode_info,
        mode_plotting,
        loading_info,
    ) = _my_solve_parser.read_args()
    if TYPE_CHECKING:
        save_info: 'SaveInfo | None'
        mode_info: 'ModeInfo'
        mode_plotting: 'PlotInfo'
        loading_info: 'LoadInfo'
    # adjust root logger level, if needed for verbosity/debug level,
    # and pop the verbosity / debug arguments
    my_logger = adjust_root_logger_level(my_logger, mode_info)
    my_logger.info('Input arguments read.')
    # return the parsed elements
    return save_info, mode_plotting, loading_info


def _generate_and_save_plots(save_info: 'SaveInfo | None', mode_plotting: 'PlotInfo', loading_info: 'LoadInfo') -> None:
    # initialize the plotter object
    my_plotter = QuadraticAEPlotter(
        load_information=loading_info,
        plotting_dict=mode_plotting,
        save_information=save_info
    )
    # generate the requested plots
    my_plotter.generate_plots()
    # show plot(s), if requested
    if mode_plotting['show_plots']:
        plt.show()


def perform_complete_plotting_run(logger: 'Logger') -> None:
    # first retrieve data from inlist and command line
    save_info, mode_plotting, loading_info = _get_and_parse_elements(my_logger=logger)
    # make plots and save them
    _generate_and_save_plots(save_info=save_info, mode_plotting=mode_plotting, loading_info=loading_info)
