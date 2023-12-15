"""Python module containing a class that contains the necessary methods to compare adiabatic and non-adiabatic computations.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
###############################################################################
# import statements
import logging

# relative imports
from .plotting import GYREPlot, MESAPlot
from ..mode_input_generator import InputGen
from ..model_handling import ModelLoader
from ..read_load_info import InfoDirHandler
###############################################################################


# set up logger
logger = logging.getLogger(__name__)


# class that compares the non-adiabatic and adiabatic GYRE computation results
class AdNadComparer:
    """Python class containing methods that compare the non-adiabatic with the adiabatic GYRE pulsation computations results.

    Notes
    -----
    Don't forget to change the 'base_dir' variable to your output directory path!

    Parameters
    ----------
    rotation_rate_percent : int
        The rotation rate in % of critical (Roche) rotation rate.
    mode_info_object : InputGen
        The object containing information on the mode(s).
    base_dir : str or None, optional
        The path to the directory containing sub-directories with GYRE output; by default None.
    alternative_directory_path : str or None, optional
        An alternative path to the directory containing the file containing information on the information directories; by default None.
    alternative_file_name : str or None, optional
        An alternative name of the file containing information on the information directories; by default None.
    """

    _mode_info: InputGen
    _ad_base_dir: str
    _nad_base_dir: str
    _info_dict: dict
    _gyre_plotter: GYREPlot
    _mesa_plotter: MESAPlot

    # base_dir='/lhome/jordanv/Documents/Nonlinear/GYRE_Lee_star_output/'
    def __init__(
        self,
        sys_args,
        rotation_rate_percent,
        mode_info_object,
        base_dir=None,
        alternative_directory_path=None,
        alternative_file_name=None,
        ad_path=None,
        nad_path=None,
    ):
        # default paths
        if ad_path is None:
            ad_path = f'output_rot_{rotation_rate_percent}'
        if nad_path is None:
            nad_path = f'output_nad_rot_{rotation_rate_percent}'
        # store the mode information object
        self._mode_info = mode_info_object
        # create specific non-adiabatic and adiabatic base dirs
        logger.debug('Generating base dir paths.')
        self._ad_base_dir = f'{base_dir}{ad_path}/'
        self._nad_base_dir = f'{base_dir}{nad_path}/'
        logger.debug('Base dir paths generated.')
        # store the dictionary that contains the paths to
        # the information directories
        _reader = InfoDirHandler(
            sys_arguments=sys_args,
            alternative_directory_path=alternative_directory_path,
            alternative_file_name=alternative_file_name,
        )
        _reader.parse_info_directories()
        self._info_dict = _reader.gyre_comparison_info

    # load a triplet of nonradial modes
    def load_modes(self, radial_order_combo_nr=0):
        """Loads the necessary information from a triplet of nonradial modes.

        Parameters
        ----------
        radial_order_combo_nr : int, optional
            The index number of the set of different combinations of radial orders.
    ; by default 0.
        """
        logger.debug('Reading mode information.')
        # construct a common kwargs dictionary
        _common_kwargs = {
            **self._info_dict,
            'infer_dir': False,
            'search_l': self._mode_info.mode_l,
            'search_m': self._mode_info.mode_m,
            'search_n': self._mode_info.combinations_radial_orders[
                radial_order_combo_nr
            ],
        }
        # load models
        _mods_ad = ModelLoader(
            gyre_output_dir=self._ad_base_dir, **_common_kwargs
        )
        _mods_ad.read_data_files(
            gyre_detail=True, mesa_profile=True, g_modes=[True, True, True]
        )
        _mods_nad = ModelLoader(
            gyre_output_dir=self._nad_base_dir, **_common_kwargs
        )
        _mods_nad.read_data_files(
            gyre_detail=True, mesa_profile=True, g_modes=[True, True, True]
        )
        logger.debug('Mode information read.')
        # store the GYRE information in the GYRE plotting object
        self._gyre_plotter = GYREPlot(
            gyre_ad=_mods_ad.gyre_details, gyre_nad=_mods_nad.gyre_details
        )
        # store the MESA information in the MESA plotting object
        self._mesa_plotter = MESAPlot(mesa=_mods_ad.mesa_profile)
        logger.debug('Mode information stored.')

    # generate GYRE mode eigenfunction comparison figures
    def generate_mode_eigenfunction_comparison_figures(
        self, radius_profiles=False
    ):
        """Method generating mode eigenfunction comparison figures.

        Parameters
        ----------
        radius_profiles : bool, optional
            If True, plot GYRE profiles in function of the normalized radius. If False, plot GYRE profiles in function of the normalized mass coordinate.
            Default: False.
        """
        logger.debug('Generating mode eigenfunction comparison figures.')
        # generate x-quantity dictionary based on boolean input
        _x_quantity_dict = (
            {'x_quantity': 'x', 'x_label': r'r / R$_*$'}
            if radius_profiles
            else {'x_quantity': 'mass', 'x_label': r'm / M$_*$'}
        )
        # generate x-bounds dictionary
        _x_bound_dict = {'left': 0.0, 'right': 1.0}
        # create the figures
        for _mnr in range(1, 4):
            # generate the comparison figure of the real and imaginary radial eigenfunctions
            self._gyre_plotter.figure_comparison(
                **_x_quantity_dict,
                y_label=[r'$Re[\xi_R]$', r'$Im[\xi_R]$'],
                mode_nr=_mnr,
                x_bounds=_x_bound_dict,
                real_quantity_y=[True, False],
            )
            # generate the comparison figure of the real and imaginary horizontal eigenfunctions
            self._gyre_plotter.figure_comparison(
                y_label=[r'$Re[\xi_H]$', r'$Im[\xi_H]$'],
                mode_nr=_mnr,
                y_quantity='xi_h',
                x_bounds=_x_bound_dict,
                **_x_quantity_dict,
                real_quantity_y=[True, False],
            )
        logger.debug('Mode eigenfunction comparison figures generated.')

    # generate MESA profile figures
    def generate_profile_figures(self, radius_profiles=False):
        """Method generating MESA profile figures for the stellar evolution model used for the mode computations.

        Parameters
        ----------
        radius_profiles : bool, optional
            If True, plot MESA profiles in function of the normalized radius. If False, plot MESA profiles in function of the normalized mass coordinate.
            Default: False.
        """
        logger.debug('Generating MESA profile figures.')
        # generate x-quantity dictionary based on boolean input
        if radius_profiles:
            _x_quantity_dict = {
                'x_quantity': 'radius',
                'x_label': r'r / R$_*$',
                'x_normalizing_quantity': (max, 'radius'),
            }
        else:
            _x_quantity_dict = {
                'x_quantity': 'mass',
                'x_label': r'm / M$_*$',
                'x_normalizing_quantity': (max, 'mass'),
            }
        # generate x-bounds dictionary
        _x_bound_dict = {'left': 0.0, 'right': 1.0}
        # create the MESA profile figures
        # - H and He profiles
        self._mesa_plotter.figure_profile(
            y_quantity=['h1', 'h2', 'he3', 'he4'],
            y_legend_labels=['h1', 'h2', 'he3', 'he4'],
            y_label=r'x',
            **_x_quantity_dict,
            x_bounds=_x_bound_dict,
            mode_nr=0,
        )
        # - brunt-vaisala frequency figure
        self._mesa_plotter.figure_profile(
            y_quantity=[
                'brunt_N2',
                'brunt_N2_structure_term',
                'brunt_N2_composition_term',
            ],
            y_legend_labels=[
                r'N$^2$',
                r'N$^2_{structure}$',
                r'N$^2_{composition}$',
            ],
            y_label=r'N$^2$',
            **_x_quantity_dict,
            x_bounds=_x_bound_dict,
            mode_nr=0,
        )
        # - lamb frequency figure
        self._mesa_plotter.figure_profile(
            y_quantity='lamb_S2',
            y_legend_labels=None,
            y_label=r'S$^2$',
            **_x_quantity_dict,
            x_bounds=_x_bound_dict,
            mode_nr=0,
        )
        # - sound speed figure
        self._mesa_plotter.figure_profile(
            y_quantity='csound',
            y_legend_labels=None,
            y_label=r'c$_{s}$',
            **_x_quantity_dict,
            x_bounds=_x_bound_dict,
            mode_nr=0,
        )
        # - hydrostatic equilibrium figure
        self._mesa_plotter.figure_profile(
            y_quantity=['hyeq_lhs', 'hyeq_rhs'],
            y_legend_labels=None,
            quantity_diff=True,
            y_label=r'lhs - rhs',
            **_x_quantity_dict,
            x_bounds=_x_bound_dict,
            mode_nr=0,
        )
        # - gamma 1 derivative profile figure
        self._mesa_plotter.figure_profile(
            y_quantity='gamma_1_adDer',
            y_legend_labels=None,
            y_label=r'(d$\Gamma_1$/dr)$_S$',
            **_x_quantity_dict,
            x_bounds=_x_bound_dict,
            mode_nr=0,
        )
        # - Dmix profile figure
        self._mesa_plotter.figure_profile(
            y_quantity=[
                'log_D_mix',
                'log_D_conv',
                'log_D_ovr',
                'log_D_minimum',
            ],
            y_legend_labels=[
                r'log_D_mix',
                r'log_D_conv',
                r'log_D_ovr',
                r'log_D_min',
            ],
            y_label=r'log D',
            **_x_quantity_dict,
            x_bounds=_x_bound_dict,
            mode_nr=0,
        )
        # - pressure profile figure
        self._mesa_plotter.figure_profile(
            y_quantity='pressure',
            y_legend_labels=None,
            y_label=r'P',
            **_x_quantity_dict,
            x_bounds=_x_bound_dict,
            mode_nr=0,
        )
        # - density profile figure
        self._mesa_plotter.figure_profile(
            y_quantity='logRho',
            y_legend_labels=None,
            y_label=r'log $\rho$',
            **_x_quantity_dict,
            x_bounds=_x_bound_dict,
            mode_nr=0,
        )
        # - temperature profile figure
        self._mesa_plotter.figure_profile(
            y_quantity='logT',
            y_legend_labels=None,
            y_label=r'log T',
            **_x_quantity_dict,
            x_bounds=_x_bound_dict,
            mode_nr=0,
        )
        # - thermodynamic gradient profile figure
        self._mesa_plotter.figure_profile(
            y_quantity=['grada', 'gradT', 'gradr'],
            y_legend_labels=[r'$\Delta_a$', r'$\Delta_T$', r'$\Delta_r$'],
            y_label=r'$\Delta$',
            **_x_quantity_dict,
            x_bounds=_x_bound_dict,
            mode_nr=0,
        )
        # possible keys:
        """dict_keys(['gamma_1_adDer', 'h1', 'h2', 'he3', 'he4', 'c12', 'c13',
        'n14', 'o16', 'o17', 'ne20', 'ne22', 'mg24', 'al27', 'si28', 's32', 'ar36',
        'brunt_N2', 'brunt_N2_structure_term', 'brunt_N2_composition_term', 'lamb_S',
        'lamb_S2', 'hyeq_lhs', 'hyeq_rhs', 'x', 'y', 'z', 'mlt_mixing_length',
        'log_D_mix', 'log_D_conv', 'log_D_ovr', 'log_D_minimum', 'conv_mixing_type',
        'zone', 'logT', 'logRho', 'luminosity', 'velocity', 'entropy', 'csound', 'mu',
        'q', 'radius', 'tau', 'pressure', 'opacity', 'eps_nuc', 'non_nuc_neu',
        'log_conv_vel', 'mass', 'mmid', 'grada', 'gradT', 'gradr',
        'pressure_scale_height', 'grav', 'center_h1', 'center_he3', 'center_he4',
        'center_c12', 'center_n14', 'center_o16', 'center_ne20', 'center_eta',
        'star_mass_h1', 'star_mass_he3', 'star_mass_he4', 'star_mass_c12',
        'star_mass_n14', 'star_mass_o16', 'star_mass_ne20', 'he_core_mass',
        'c_core_mass', 'o_core_mass', 'si_core_mass', 'fe_core_mass',
        'neutron_rich_core_mass', 'tau10_mass', 'tau10_radius', 'tau100_mass',
        'tau100_radius', 'initial_z', 'initial_mass', 'photosphere_L',
        'photosphere_r', 'Teff', 'dynamic_time', 'kh_timescale', 'nuc_timescale',
        'power_nuc_burn', 'power_h_burn', 'power_he_burn', 'power_neu',
        'model_number', 'num_zones', 'star_age', 'time_step', 'burn_min1',
        'burn_min2', 'time_seconds', 'version_number', 'compiler', 'build',
        'MESA_SDK_version', 'date'])
        """


###############################################################################
