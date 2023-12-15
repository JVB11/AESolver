"""Python run script used to compute the quadratic coupling coefficients and generate the (stationary) solutions to the quadratic amplitude equations. Also computes the coupling coefficient profile and generates visualizations of this profile and the integral kernels used to compute the coupling coefficient.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import sys

# import custom mode-coupling-computing modules
from aesolver import NumericalSaver, QuadraticAEGridSolver, QuadraticAEProfileSolver

# import custom generic parser and log formatter modules
from generic_parser import GenericParser
from log_formatter import adjust_root_logger_level
from path_resolver import resolve_path_to_file

# import local modules
from .data_modules import check_save_file_name

# type checking imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any
    import argparse
    from logging import Logger
    
    # import specific typed dictionaries
    from .data_modules import ModeInfoOrPolyDict, ModeSelectionDict, ModeComputationInfoDict, ComputationDict, SavingDict, PolyModelDict, AngularIntegrationDict, CouplingCoefficientProfileSelectionDict, CouplingCoefficientProfilePlotDict


# subclass the GenericParser superclass for use in solving the quadratic amplitude equations
class QuadraticSolveParser(GenericParser):
    """Subclass of the GenericParser superclass used to parse arguments
    for solving the quadratic amplitude equations.

    Parameters
    ----------
    full_inlist_path : Path
        The path to the toml inlist that will be read.   
    """
    
    # attribute type declarations
    _parser: 'argparse.ArgumentParser'
    _data_read_dictionary: 'None | dict[str, Any]'
    _inlist_suffix: str
    _inlist_name: str
    _base_dir: str
    _inlist_dir: str
    _args: 'argparse.Namespace'

    def __init__(
        self,
        full_inlist_path: 'Path',
    ) -> None:
        super().__init__(
            base_directory_name='AESolver',
            full_inlist_path=full_inlist_path,
            base_dir=None,
            inlist_name=None,
            inlist_dir=None,
            inlist_suffix=None,
        )

    # overload the read_specific_arguments method
    def _set_up_specific_argparser(self) -> None:
        """Overloaded method used to set up a limited amount of parser groups and arguments.
        The majority of all solve run configuration items are passed through an inlist.
        """
        # set up file information input
        file_info_group = self._parser.add_argument_group(
            'File information',
            'Group containing arguments that specify information about GYRE files.',
        )
        # store the save file name (used to save your HDF5 data output files)
        file_info_group.add_argument(
            '--save_file_name',
            type=str or None,
            default=None,
            help='Save name for the HDF5 file.',
        )

    def get_save_name(self) -> str:
        """Retrieves the file save name from parser arguments.

        Returns
        -------
        str
            File save name.
        """
        return self._args.save_file_name
        
    # convert specific toml input from list to tuple data
    def _convert_toml_input_list_to_tuple(self) -> None:
        """Converts specific toml input lists to tuples, where necessary.
        """
        # store path dicts to list arguments that need to be converted
        _conversion_path_lists = [['coupling_coefficient', 'terms_cc_conj']]
        # perform the conversion inplace
        for _l in _conversion_path_lists:
            _val: list = self._data_read_dictionary[_l[0]][_l[1]]
            self._data_read_dictionary[_l[0]].update({_l[1]: tuple(_val)})

    # overload the read arguments method for toml inlists
    def read_toml_args(self) -> 'tuple[SavingDict, ModeInfoOrPolyDict, CouplingCoefficientProfileSelectionDict, CouplingCoefficientProfilePlotDict, ModeSelectionDict, ModeComputationInfoDict, ComputationDict, ModeInfoOrPolyDict, PolyModelDict, AngularIntegrationDict, str, str, str, bool]':
        """Read the input arguments from the toml-style inlist.

        Returns
        -------
        my_save_dict : SavingDict
            Contains information necessary to save the mode data (directories, specifier, format).
        my_mode_info_dict : ModeInfoOrPolyDict
            Contains generic information of the modes (rotation percentage, directories in which data files are stored, debug/verbose options).
        my_coupling_coefficient_selection_dict : CouplingCoefficientProfileSelectionDict
            Contains information on the parameters used to determine compute behavior when generating coupling coefficient profiles.
            (Mode selection for grid computations is handled by 'my_mode_selection_dict').
        my_coupling_coefficient_profile_plot_dict : CouplingCoefficientProfilePlotDict
            Contains information that determines the plotting behavior for the coupling coefficient profiles
            (these are for now not saved to disk; whether you wish to view individual integral kernels and the term-by-term contributions of the coupling coefficient).
        my_mode_selection_dict : ModeSelectionDict
            Contains information on the mode selection for the computation of grids.
            (Mode selection for coupling coefficient profile computation is handled by 'my_coupling_coefficient_selection dict').
        my_mode_computation_info_dict : ModeComputationInfoDict
            Contains the necessary info for the mode (coupling) computation.
        my_computation_dict : ComputationDict
            Contains generic information about computational behavior (whether you are computing a grid or a profile and want to show progress bars).
        my_mode_poly_dict : ModeInfoOrPolyDict
            Contains generic information of the polytropic modes (rotation percentage, directories in which data files are stored, debug/verbose options).
        my_poly_model_dict : PolyModelDict
            Contains information on the (GYRE) polytropic structure models that need to be loaded (if necessary).
        my_ang_integ_dict : AngularIntegrationDict
            Contains information on how to compute the angular integrals.
        str
            Adiabatic specifier for (GYRE) mode data gathering.
        str
            Nonadiabatic specifier for (GYRE) mode data gathering.
        str
            Mode specifier for (GYRE) polytrope data gathering.
        bool
            Whether debug information should be printed.
        """
        # convert specific toml input lists to tuples (where necessary)!
        self._convert_toml_input_list_to_tuple()
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
        # generate the dictionary that will hold the arguments that select polytropic mode and structure files
        my_mode_poly_dict = self.read_in_to_dict(
            [
                ('rot_pct', ['rotation', 'rot_percent'], False),
                ('base_dir', ['POLY_selection', 'mode_path'], False),
                (
                    'alternative_directory_path',
                    ['POLY_selection', 'alternative', 'directory_path'],
                    False,
                ),
                (
                    'alternative_file_name',
                    ['POLY_selection', 'alternative', 'file_name'],
                    False,
                ),
                ('verbose', ['verbosity-debug', 'verbose'], False),
                ('debug', ['verbosity-debug', 'debug'], False),
            ]
        )
        # generate the dictionary that holds information on the mode selection
        my_mode_selection_dict = self.read_in_to_dict(
            [
                ('driven_n_low', ['mode_info', 'parent', 'n', 'low']),
                ('driven_n_high', ['mode_info', 'parent', 'n', 'high']),
                ('damped_n_low1', ['mode_info', 'daughter_1', 'n', 'low']),
                ('damped_n_high1', ['mode_info', 'daughter_1', 'n', 'high']),
                ('damped_n_low2', ['mode_info', 'daughter_2', 'n', 'low']),
                ('damped_n_high2', ['mode_info', 'daughter_2', 'n', 'high']),
            ]
        )
        index_list = ['parent', 'daughter_1', 'daughter_2']
        my_mode_selection_dict['mode_l'] = self.read_in_to_list(
            [['mode_info', i, 'l'] for i in index_list]
        )
        my_mode_selection_dict['mode_m'] = self.read_in_to_list(
            [['mode_info', i, 'm'] for i in index_list]
        )
        # generate the dictionary that holds computation details
        my_mode_computation_info_dict = self.read_in_to_dict(
            [
                ('adiabatic', ['coupling_coefficient', 'adiabatic'], False),
                ('use_complex', ['coupling_coefficient', 'use_complex'], False),
                (
                    'use_symbolic_profiles',
                    ['coupling_coefficient', 'symbolic_profiles'],
                    False,
                ),
                (
                    'use_brunt_def',
                    ['coupling_coefficient', 'brunt_def_profiles'],
                    False,
                ),
                (
                    'use_rotating_formalism',
                    ['rotation', 'rotating_formalism'],
                    False,
                ),
                (
                    'terms_cc_conj',
                    ['coupling_coefficient', 'terms_cc_conj'],
                    False,
                ),
                (
                    'polytrope_comp',
                    ['coupling_coefficient', 'compute_for_polytrope'],
                    False,
                ),
                (
                    'analytic_polytrope',
                    ['coupling_coefficient', 'analytic_polytrope'],
                    False,
                ),
            ]
        )
        # generate the dictionary that holds computation details
        my_computation_dict = self.read_in_to_dict(
            [
                (
                    'use_parallel',
                    ['computation', 'parallel_mode_coupling'],
                    False,
                ),
                (
                    'compute_grid',
                    ['computation', 'compute_grid_models'],
                    False,
                ),
                (
                    'compute_profile',
                    ['computation', 'compute_coefficient_profile'],
                    False,
                ),
                (
                    'progress_bars',
                    ['computation', 'display_progress'],
                    False,
                ),
            ]
        )
        # generate the dictionary that holds save options
        my_save_dict = self.read_in_to_dict(
            [
                ('base_dir', ['saving', 'solve', 'base_dir'], False),
                ('save_dir', ['saving', 'solve', 'save_dir'], False),
                (
                    'extra_specifier',
                    ['saving', 'solve', 'extra_specifier'],
                    False,
                ),
                (
                    'initial_save_format',
                    ['saving', 'solve', 'save_format'],
                    False,
                ),
            ]
        )
        # generate the dictionary that holds polytrope model arguments
        my_poly_model_dict = self.read_in_to_dict(
            [
                ('polytrope_mass', ['POLY_model', 'mass'], False),
                ('polytrope_radius', ['POLY_model', 'radius'], False),
            ]
        )
        # generate the dictionary that holds angular integration kwargs
        my_ang_integ_dict = self.read_in_to_dict(
            [
                (
                    'numerical_integration_method',
                    ['computation', 'integration_method'],
                    False,
                ),
                (
                    'use_cheby_integration',
                    ['computation', 'angular', 'use_cheby'],
                    False,
                ),
                (
                    'cheby_order_multiplier',
                    ['computation', 'angular', 'cheby_mul'],
                    False,
                ),
                (
                    'cheby_blow_up_protection',
                    ['computation', 'angular', 'blow_up_protect'],
                    False,
                ),
                (
                    'cheby_blow_up_factor',
                    ['computation', 'angular', 'blow_up_fac'],
                    False,
                ),
            ]
        )
        # generate the mode coupling coefficient profile selection dictionary
        my_coupling_coefficient_selection_dict = self.read_in_to_dict(
            [
                ('parent_n', ['mode_info', 'coupling', 'parent', 'n'], False),
                ('daughter_1_n', ['mode_info', 'coupling', 'daughter_1', 'n'], False),
                ('daughter_2_n', ['mode_info', 'coupling', 'daughter_2', 'n'], False),
            ]
        )
        my_coupling_coefficient_selection_dict['mode_l'] = self.read_in_to_list(
            [['mode_info', i, 'l'] for i in index_list]
        )
        my_coupling_coefficient_selection_dict['mode_m'] = self.read_in_to_list(
            [['mode_info', i, 'm'] for i in index_list]
        )
        # generate the cc profile plot dict
        my_coupling_coefficient_profile_plot_dict = self.read_in_to_dict(
            [
                ('view_kernels', ['plotting', 'profile', 'show_kernels'], False),
                ('view_term_by_term', ['plotting', 'profile', 'show_term_by_term'], False),
            ]
        )
        # return the information dictionaries
        return (
            my_save_dict,
            my_mode_info_dict,
            my_coupling_coefficient_selection_dict,
            my_coupling_coefficient_profile_plot_dict,
            my_mode_selection_dict,
            my_mode_computation_info_dict,
            my_computation_dict,
            my_mode_poly_dict,
            my_poly_model_dict,
            my_ang_integ_dict,
            *self.read_in_to_list(
                [
                    ['GYRE_selection', 'specifier', 'adiabatic'],
                    ['GYRE_selection', 'specifier', 'nonadiabatic'],
                    ['POLY_selection', 'specifier', 'modes'],
                    ['verbosity-debug', 'get_debug_info'],
                ]
            ),
        )


def _get_path_to_inlist(
    sys_arguments: list,
    inlist_file_name: str = 'inlist',
    inlist_file_suffix: str = 'toml',
) -> 'Path':
    """Generates the path to the inlist to be read.
    
    Notes
    -----
    Expects either the command line argument '--save_file_name' (with corresponding save file name) to be passed, or the custom inlist file name specifiers: file_name, inlist_file_name, inlist_suffix, relative_inlist_path, followed by the command line argument '--save_file_name' (with corresponding save file name).

    Parameters
    ----------
    sys_arguments : list
        _description_
    inlist_file_name : str, optional
        _description_, by default 'inlist'
    inlist_file_suffix : str, optional
        _description_, by default 'toml'

    Returns
    -------
    Path
        Pathlib Path to the inlist file.

    Raises
    ------
    RuntimeError
        Raised if a wrong number of command line input arguments is passed.
    """
    # generate relevant properties used to create the path to the inlist file
    match len(sys_arguments):
        case 6:
            # ARGUMENTS:
            # 0) file name
            # 1) save file name
            # 2) save file suffix
            # 3) relative inlist path
            # 4) --save_file_name
            # 5) the name of the save file (not used for profiles)
            # check if the fourth actual argument is the save file name specifier
            check_save_file_name(sys_arguments=sys_arguments, pos=4)
            # construct file name from command line input arguments
            full_file_name = f'{sys_arguments[1]}.{sys_arguments[2]}'
            # store the relative path to the inlist directory from this file
            relative_inlist_path = f'{sys_arguments[3]}'
        case 3:
            # ARGUMENTS:
            # 0) file name
            # 1) --save_file_name
            # 2) the name of the save file (not used for profiles)
            # check if the first actual argument is the save file name specifier
            check_save_file_name(sys_arguments=sys_arguments, pos=1)
            # YOU ARE USING THE DEFAULT INLIST:
            # construct full file name from default arguments to this function (i.e. use the default inlist)
            full_file_name = f'{inlist_file_name}.{inlist_file_suffix}'
            # store the relative path to the inlist directory from this file
            relative_inlist_path = './inlists/'
        case _:
            raise RuntimeError(f'A number of input arguments ({len(sys_arguments)}) was passed to this script for which the behavior is undefined.\n\nPlease either specify:\n*) inlist file name, inlist file suffix and relative path to the inlist file from the repository base directory, followed by "--save_file_name xxx" (to read a custom inlist)\n*) "--save_file_name xxx" only (to use the default inlist).\n\nOther options have not yet been implemented.')
    # now resolve the path to the inlist and return it
    return resolve_path_to_file(
        file_name=full_file_name,
        default_path=relative_inlist_path,
        sys_arguments=sys_arguments,
    )
    
    
def _get_and_parse_elements(my_logger: 'Logger') -> 'tuple[bool, QuadraticSolveParser, SavingDict, ModeInfoOrPolyDict, CouplingCoefficientProfileSelectionDict, CouplingCoefficientProfilePlotDict, ModeSelectionDict, ModeComputationInfoDict, ComputationDict, ModeInfoOrPolyDict, PolyModelDict, AngularIntegrationDict, str, str, str, bool, str]':
    """Retrieves information passed with the run inlist and the command line.

    Parameters
    ----------
    my_logger : Logger
        The root logging object.

    Returns
    -------
    use_toml : bool
        Whether a toml inlist was read. (Should be True)
    my_solve_parser : QuadraticSolveParser
        The argument parser specific for this parsing action.
    save_info : SavingDict
        Contains information necessary to save the mode data (directories, specifier, format).
    mode_info : ModeInfoOrPolyDict
        Contains generic information of the modes (rotation percentage, directories in which data files are stored, debug/verbose options).
    cc_profile_selection : CouplingCoefficientProfileSelectionDict
        Contains information on the parameters used to determine compute behavior when generating coupling coefficient profiles. (Mode selection for grid computations is handled by 'my_mode_selection_dict').
    cc_profile_plot : CouplingCoefficientProfilePlotDict
        Contains information that determines the plotting behavior for the coupling coefficient profiles (these are for now not saved to disk; whether you wish to view individual integral kernels and the term-by-term contributions of the coupling coefficient).
    mode_selection : ModeSelectionDict
        Contains information on the mode selection for the computation of grids. (Mode selection for coupling coefficient profile computation is handled by 'my_coupling_coefficient_selection dict').
    mode_computation : ModeComputationInfoDict
        Contains the necessary info for the mode (coupling) computation.
    comp : ComputationDict
        Contains generic information about computational behavior (whether you are computing a grid or a profile and want to show progress bars).
    poly_select : ModeInfoOrPolyDict
        Contains generic information of the polytropic modes (rotation percentage, directories in which data files are stored, debug/verbose options).
    poly_model : PolyModelDict
        Contains information on the (GYRE) polytropic structure models that need to be loaded (if necessary).
    a_i : AngularIntegrationDict
        Contains information on how to compute the angular integrals.
    ad : str
        Adiabatic specifier for (GYRE) mode data gathering.
    nad : str
        Nonadiabatic specifier for (GYRE) mode data gathering.
    poly_modes : str
        Mode specifier for (GYRE) polytrope data gathering.
    get_debug_info : bool
        Whether debug information should be printed.
    fs_name : str
        The file save name.
    """
    # check if any command line arguments are passed, and retrieve inlist name and directory
    inlist_path = _get_path_to_inlist(sys_arguments=sys.argv)
    # check if you are using a toml file
    use_toml = inlist_path.suffix == '.toml'
    # initialize the parser object
    my_solve_parser = QuadraticSolveParser(full_inlist_path=inlist_path)
    # read the input arguments
    (
        save_info,
        mode_info,
        cc_profile_selection,
        cc_profile_plot,
        mode_selection,
        mode_computation,
        comp,
        poly_select,
        poly_model,
        a_i,
        ad,
        nad,
        poly_modes,
        get_debug_info,
    ) = my_solve_parser.read_args()
    # retrieve the file save name
    fs_name = my_solve_parser.get_save_name()
    if fs_name is None:
        my_logger.error(
            'No save file name specified! Now exiting before computations.'
        )
        sys.exit()
    # return the parser and the parsed information
    return use_toml, my_solve_parser, save_info, mode_info, cc_profile_selection, cc_profile_plot, mode_selection, mode_computation, comp, poly_select, poly_model, a_i, ad, nad, poly_modes, get_debug_info, fs_name


def _perform_computations(comp: 'ComputationDict', mode_computation: 'ModeComputationInfoDict', poly_select: 'ModeInfoOrPolyDict', poly_model: 'PolyModelDict', a_i: 'AngularIntegrationDict', mode_selection: 'ModeSelectionDict', poly_modes: str, my_solve_parser: 'QuadraticSolveParser', use_toml: bool, get_debug_info: bool, mode_info: 'ModeInfoOrPolyDict', ad: str, nad: str, cc_profile_selection: 'CouplingCoefficientProfileSelectionDict', cc_profile_plot: 'CouplingCoefficientProfilePlotDict', fs_name: str, my_logger: 'Logger') -> 'QuadraticAEGridSolver | QuadraticAEProfileSolver':
    # initialize the solver object
    if comp['compute_grid'] and comp['compute_profile']:
        raise RuntimeError('Please choose between the two run options: "compute_grid" and "compute_profile"!\nBoth actions cannot be undertaken within a single run, yet, both were set to True at the start of this run.')
    elif comp['compute_grid']:
        # -- GRID 
        # - polytrope computations
        if mode_computation['polytrope_comp']:
            # adjust root logger level, if needed for verbosity/debug level, and pop the verbosity / debug arguments
            my_logger = adjust_root_logger_level(my_logger, poly_select)
            my_logger.info('Input arguments read.')
            # solver init
            my_solver = QuadraticAEGridSolver(
                sys.argv,
                **poly_select,
                **mode_computation,
                **poly_model,
                **a_i,
                mode_selection_dict=mode_selection,
                ad_path=poly_modes,
                nad_path=None,
                inlist_path=my_solve_parser.inlist_path,
                toml_use=use_toml,
                use_parallel=comp['use_parallel'],
                get_debug_info=get_debug_info,
                progress_bars=comp['progress_bars'],
            )
        # - MESA + GYRE computations
        else:
            # adjust root logger level, if needed for verbosity/debug level, and pop the verbosity / debug arguments
            my_logger = adjust_root_logger_level(my_logger, mode_info)
            my_logger.info('Input arguments read.')
            # solver init
            my_solver = QuadraticAEGridSolver(
                sys.argv,
                **mode_info,
                **mode_computation,
                **poly_model,
                **a_i,
                mode_selection_dict=mode_selection,
                ad_path=ad,
                nad_path=nad,
                inlist_path=my_solve_parser.inlist_path,
                toml_use=use_toml,
                use_parallel=comp['use_parallel'],
                get_debug_info=get_debug_info,
                progress_bars=comp['progress_bars'],
            )
    elif comp['compute_profile']:
        # -- PROFILE
        # profile computations for specific triad
        # adjust root logger level, if needed for verbosity/debug level, and pop the verbosity / debug arguments
        my_logger = adjust_root_logger_level(my_logger, mode_info)
        my_logger.info('Input arguments read.')
        # solver init
        my_solver = QuadraticAEProfileSolver(
            sys.argv,
            **mode_info,
            **mode_computation,
            **poly_model,
            **a_i,
            cc_profile_selection_dict=cc_profile_selection,
            ad_path=ad,
            nad_path=nad,
            inlist_path=my_solve_parser.inlist_path,
            toml_use=use_toml,
            use_parallel=comp['use_parallel'],
            get_debug_info=get_debug_info,
            save_name=fs_name,
            **cc_profile_plot,
            progress_bars=comp['progress_bars'],
        )
    else:
        raise RuntimeError('No action selected from the two available actions: "compute_grid" or "compute_profile".')
    # COMPUTE the requested information
    my_solver()
    # return the solver object
    return my_solver


def _save_data(comp: 'ComputationDict', fs_name: str, mode_computation: 'ModeComputationInfoDict', save_info: 'SavingDict', my_solver: 'QuadraticAEGridSolver | QuadraticAEProfileSolver') -> None:
    # save the output from the solver by initializing the saver object and running its save method
    # - CHECK if you have generated any grid data, and save it
    if comp['compute_grid']:
        if TYPE_CHECKING:
            assert isinstance(my_solver, QuadraticAEGridSolver)
        # -- SAVE the grid data
        my_saver = NumericalSaver(
            save_name=fs_name,
            solving_object=my_solver,
            polytropic=mode_computation['polytrope_comp'],
            nr_modes=3,
            **save_info,
        )
        my_saver()


def perform_complete_solve_run(logger: 'Logger') -> None:
    """Performs the necessary solving actions according to the parsed inlist and command line arguments.
    """
    # first retrieve data from inlist and command line
    use_toml, my_solve_parser, save_info, mode_info, cc_profile_selection, cc_profile_plot, mode_selection, mode_computation, comp, poly_select, poly_model, a_i, ad, nad, poly_modes, get_debug_info, fs_name = _get_and_parse_elements(my_logger=logger)
    # use the retrieved data to run the requested computations (if possible)
    # and store the data in a solver object
    my_solve_object = _perform_computations(comp=comp, mode_computation=mode_computation, poly_select=poly_select, poly_model=poly_model, mode_selection=mode_selection, poly_modes=poly_modes, my_solve_parser=my_solve_parser, use_toml=use_toml, get_debug_info=get_debug_info, mode_info=mode_info, ad=ad, nad=nad, cc_profile_selection=cc_profile_selection, cc_profile_plot=cc_profile_plot, fs_name=fs_name, a_i=a_i, my_logger=logger)
    # save the computed data
    _save_data(comp=comp, fs_name=fs_name, mode_computation=mode_computation, save_info=save_info, my_solver=my_solve_object)
