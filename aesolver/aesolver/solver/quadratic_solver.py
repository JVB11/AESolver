"""Python package containing the class that will handle the solving of the quadratic amplitude equations.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
###############################################################################
############################# import statements ###############################
###############################################################################
import logging
import sys
import typing
import numpy as np
import matplotlib.pyplot as plt
from astropy.constants import M_sun, R_sun  # type: ignore (although the type checker does not find it, R_sun is part of astropy.constants)
from pathlib import Path
from tqdm import tqdm

# relative imports
# - used to load mode combination data
from ..mode_input_generator import InputGen

# - used to select directories
from ..read_load_info import InfoDirHandler

# - computes stationary solutions for A = B + C
from ..stationary_solving import ThreeModeStationary

# - hyperbolicity checking functions
from ..stability_checker import check_hyper, hyper_jac, hyper_jac_three

# - used to perform pre-checks
from ..pre_checks import PreCheckQuadratic, PreThreeQuad

# - used to handle frequency (GYRE) data
#   and to compute the quadratic coupling coefficient
from ..frequency_handling import FH
from ..coupling_coefficients import QCCR

# - used to load more data
from ..model_handling import ModelLoader

# - used to plot radial kernel profiles
from .profile_helper_classes import RadialKernelData, AngularKernelData

# import custom modules
# - computes disc integral factors.
from disc_integrator import di  # type: ignore (CPP-interfacing module)

# - used to convert GYRE data to numpy array(s)
from carr_conv import re_im

# - poll user
from .profile_helper_classes import poll_user_profile_for_combination_number

# - plot profiles
from .profile_plot_classes import (
    get_model_dependent_part_save_name,
    get_figure_output_base_path,
    save_fig,
    double_masking_integrated_radial_profile_figure_template_function,
    single_integrated_radial_profile_figure_template_function,
    create_angular_contribution_figures_for_specific_contribution,
    create_radial_contribution_figures_for_specific_contribution,
)

# - print information for plotting
from .profile_print_functions import (
    print_normed_cc_and_get_adjust_ratio,
    print_check_hyperbolic,
    print_surface_luminosities,
    print_theoretical_amplitudes,
    display_contributions,
    print_cc_contribution_info,
)

# type checking imports
if typing.TYPE_CHECKING:
    import numpy.typing as npt
    from typing import Literal
    from collections.abc import Sequence

###############################################################################


# set up logger
logger = logging.getLogger(__name__)


# use this adjusted fontset for every figure (obtain correct epsilon)
plt.rcParams['mathtext.fontset'] = 'cm'


# define custom exception classes
class PhysicsError(Exception):
    def __init__(self, msg: str) -> None:
        super().__init__(msg)


class InlistError(Exception):
    def __init__(self, msg: str, inlist_path: str | None) -> None:
        inlist_error_message = f'{msg}\n(Path to inlist file: {inlist_path})'
        super().__init__(inlist_error_message)


# class that handles the solving of the amplitude equations
class QuadraticAESolver:
    """The class that handles the solving of the quadratic amplitude equations for gravity-mode pulsating stars.

    Parameters
    ----------
    rot_pct : int, optional
        Percent of (Roche) critical rotation rate, used to select the specific GYRE output files; by default 20.
    use_complex : bool, optional
        If True, use complex-valued quantities during computations. If False, use real-valued quantities during computations; by default False
    adiabatic : bool, optional
        If True, use adiabatic eigenfunctions to compute the coupling coefficients, as was done in Lee (2012). If False, use the real parts of the non-adiabatic eigenfunctions to compute the coupling coefficients; by default True
    use_rotating_formalism : bool, optional
        If True, adhere to the Lee (2012) uniform TAR rotation formalism. If False, use a non-rotating formalism (see e.g. Van Hoolst (1992)). NOTE: non-rotating formalism NOT YET implemented; by default True
    base_dir : str or None, optional
        The path to the directory containing sub-directories with GYRE output; by default None.
    alternative_directory_path : str or None, optional
        An alternative path to the directory containing the file containing information on the information directories; by default None.
    alternative_file_name : str or None, optional
        An alternative name of the file containing information on the information directories; by default None.
    use_symbolic_profiles : bool, optional
        If True, use symbolic expressions to compute most of the stellar evolution model derivatives (except for radial derivatives of the mass density of order > 1). If False, compute these derivatives numerically; by default True.
    use_brunt_def : bool, optional
        If True and 'use_symbolic_profiles' is True, use an symbolic expression in terms of the buoyancy/brunt-vaisala frequency to compute the first radial density derivative. If False, the first radial density derivative is computed numerically; by default True.
    mode_selection_dict : dict or None, optional
        The dictionary that holds information on the mode selection. If None, default information for the mode selection shall be used; by default None.
    recompute_for_profile : bool, optional
        If True, recompute the coupling coefficient, even if one was already computed. If False, check if you need to compute the coupling coefficient, but do not recompute it if it has already been computed; by default False
    """

    # attribute type declarations
    _mode_info_object: InputGen
    _nr_omp_threads: int
    _use_parallel: bool
    _numerical_integration_method: str
    _use_cheby: bool
    _cheby_order_mul: int
    _cheby_blow_up: bool
    _cheby_blow_up_fac: float
    _get_debug_info: bool
    _polytrope_comp: bool
    _analytic_polytrope: bool
    _polytrope_mass: float
    _polytrope_radius: float
    _ld_function: str
    _stat_solv: bool
    _rot_pct: int
    _use_rot_formalism: bool
    _use_symbolic_profiles: bool
    _use_brunt_def: bool
    _use_complex: bool
    _adiabatic: bool
    _terms_cc_conj: tuple[bool, bool, bool] | bool | None
    _recompute_for_profile: bool
    _ad_base_dir: str
    _nad_base_dir: str | None
    _inlist_path: str | None
    _info_dict_info_dirs: dict[str, typing.Any]
    _coupling_info: list[tuple[typing.Any, ...]]
    _coupling_stat_info: list[tuple[typing.Any, ...]]
    _stat_sol_info: list[dict[str, typing.Any] | None]
    _mode_freq_info: list[tuple[typing.Any, ...]]
    _hyperbolicity_info: list[bool]
    _save_name: str
    _poly_models: list | None
    _poly_struct: list | None
    _gyre_ad: 'list[dict[str, typing.Any]] | None'
    _gyre_nad: list | None
    _mesa: list | None
    _freq_handler: FH
    _coupler: QCCR | None
    _coupling_profile: np.ndarray | None
    _ang_mode_freqs: np.ndarray
    _driving_rates: np.ndarray
    _quality_factors: np.ndarray
    _quality_factor_products: np.ndarray
    _lag_var_surf_lum: np.ndarray
    _xir_terms_norm: np.ndarray
    _cc_profile_selection_dict: None | dict[str, bool]

    def __init__(
        self,
        sys_args: list,
        cc_profile_selection_dict: dict[str, int],
        called_from_pytest: bool = False,
        pytest_path: None | Path = None,
        rot_pct: int = 20,
        use_complex: bool = False,
        adiabatic: bool = True,
        use_rotating_formalism: bool = True,
        base_dir: str | None = None,
        alternative_directory_path: str | None = None,
        alternative_file_name: str | None = None,
        use_symbolic_profiles: bool = True,
        use_brunt_def: bool = True,
        mode_selection_dict: dict[str, typing.Any] | None = None,
        recompute_for_profile: bool = False,
        nr_omp_threads: int = 4,
        use_parallel: bool = False,
        ld_function: str = 'eddington',
        numerical_integration_method: str = 'trapz',
        use_cheby_integration: bool = False,
        cheby_order_multiplier: int = 4,
        cheby_blow_up_protection: bool = False,
        cheby_blow_up_factor: float = 1.0e3,
        stationary_solving: bool = True,
        ad_path: str | None = None,
        nad_path: str | None = None,
        inlist_path: str | None = None,
        toml_use: bool = False,
        get_debug_info: bool = False,
        terms_cc_conj: tuple[bool, bool, bool] | bool | None = None,
        polytrope_comp: bool = False,
        analytic_polytrope: bool = False,
        polytrope_mass: float = 3.0,
        polytrope_radius: float = 4.5,
        save_name: str = '',
        compute_grid: bool = False,
        compute_profile: bool = False,
    ) -> None:
        # store whether you are computing a profile or a grid
        self._compute_grid = compute_grid
        self._compute_profile = compute_profile
        # set up the mode information object
        if (not compute_grid) & (not compute_profile):
            raise RuntimeError("One of the two options 'compute_profile' or 'compute_grid' should be True when attempting to compute coupled solutions.\nNow exiting.")
        elif compute_grid & compute_profile:
            raise RuntimeError("Both of the options 'compute_profile' and 'compute_grid' are set to True.\nOnly one should be set to True when attempting to compute coupled solutions.\nNow exiting.")
        elif compute_grid:
            if mode_selection_dict is None:
                self._mode_info_object = InputGen(
                    damped_n_low1=30,
                    damped_n_high1=45,
                    driven_n_low=15,
                    driven_n_high=25,
                    mode_l=[2, 1, 1],
                    mode_m=[2, 1, 1],
                    damped_n_low2=None,
                    damped_n_high2=None,
                )
                self._cc_profile_selection_dict = None
            else:
                self._mode_info_object = InputGen(**mode_selection_dict)
                self._cc_profile_selection_dict = None
                print(self._mode_info_object.__dict__)
        elif compute_profile:
            # store the coefficient profile selection data dict
            self._cc_profile_selection_dict = cc_profile_selection_dict
            self._mode_info_object = None
        # store information for numerical computations
        self._nr_omp_threads = nr_omp_threads
        self._use_parallel = use_parallel
        self._numerical_integration_method = numerical_integration_method
        self._use_cheby = use_cheby_integration
        self._cheby_order_mul = cheby_order_multiplier
        self._cheby_blow_up = cheby_blow_up_protection
        self._cheby_blow_up_fac = cheby_blow_up_factor
        self._get_debug_info = get_debug_info
        # store whether polytropic computations are made
        self._polytrope_comp = polytrope_comp
        # store whether analytical expressions are used for
        # the polytrope model structure quantities
        self._analytic_polytrope = analytic_polytrope
        # store the polytropic mass and radius (in solar units)
        self._polytrope_mass = polytrope_mass
        self._polytrope_radius = polytrope_radius
        # store information on the chosen limb-darkening function
        self._ld_function = ld_function
        # store whether you are solving a stationary problem
        self._stat_solv = stationary_solving
        # store the rotation percentage and whether a rotating formalism is used
        self._rot_pct = rot_pct
        self._use_rot_formalism = use_rotating_formalism
        # store whether symbolic profiles need to be computed
        self._use_symbolic_profiles = use_symbolic_profiles
        # store whether a brunt-based symbolic derivative profile is to be used if the parameter 'self._use_symbolic_profiles' is True
        self._use_brunt_def = use_brunt_def
        # store whether complex integration should be used to compute the coupling coefficients or not
        self._use_complex = use_complex
        # store whether or not adiabatic eigenfunctions shall be used in the computation of the coupling coefficients
        self._adiabatic = adiabatic
        # store which modes are complex conjugated in the CC expression
        self._terms_cc_conj = terms_cc_conj
        # store whether you want to force the recompute of the coupling coefficients when generating a coupling coefficient profile
        self._recompute_for_profile = recompute_for_profile
        # create specific non-adiabatic and adiabatic base dirs
        # - default paths
        if ad_path is None:
            ad_path = f'output_rot_{rot_pct}'
        if nad_path is None:
            nad_path = f'output_nad_rot_{rot_pct}'
        logger.debug('Generating base dir paths.')
        self._ad_base_dir = f'{base_dir}{ad_path}/'
        if self._polytrope_comp:
            self._nad_base_dir = None  # no need for NAD quantities
        else:
            self._nad_base_dir = f'{base_dir}{nad_path}/'
        logger.debug('Base dir paths generated.')
        # store the inlist path
        self._inlist_path = inlist_path
        # store the dictionary that contains the paths to the information directories
        _reader = InfoDirHandler(
            sys_arguments=sys_args,
            alternative_directory_path=alternative_directory_path,
            alternative_file_name=alternative_file_name,
            toml_use=toml_use,
            inlist_path=inlist_path,
            called_from_pytest=called_from_pytest,
            pytest_file_path=pytest_path,
        )
        _reader.parse_info_directories()
        self._info_dict_info_dirs = _reader.full_info
        # drop unnecessary info
        self._info_dict_info_dirs.pop('gyre_base_dir')
        self._info_dict_info_dirs.pop('gyre_output_dir')
        # initialize the lists that will hold the necessary information
        # - mode coupling information
        if compute_grid:
            self._coupling_info = [
                tuple() for _ in self._mode_info_object.combinations_radial_orders
            ]
        elif compute_profile:
            self._coupling_info = [tuple()]
        # - mode statistics information
        self._coupling_stat_info = [tuple() for _ in self._coupling_info]
        # - stationary solution information
        self._stat_sol_info = [dict() for _ in self._coupling_info]
        # - mode frequency information
        self._mode_freq_info = [tuple() for _ in self._coupling_info]
        # - hyperbolicity (of fixed point) information
        self._hyperbolicity_info = [False for _ in self._coupling_info]
        # store a save name that is used to save figures for coupling profiles
        self._save_name = save_name

    # make the class a callable so that the requested data can be computed
    def __call__(
        self,
        compute_grid: bool = True,
        compute_profile: bool = False,
        progress_bars: bool = False,
    ) -> None:
        # depending on input, select which computation actions need to be undertaken
        # - check if there are conflicting compute options
        if compute_profile & compute_grid:
            raise InlistError(
                msg="\nYou are attempting to generate conflicting sets of data. Check if (ONLY) one of the inlist settings '[computation] - compute_grid_models' or '[computation] - compute_coefficient_profile' is set to True (if both of them are set to True, no data will be generated).",
                inlist_path=self._inlist_path,
            )
        # - plot mode coefficient profile
        elif compute_profile:
            # compute the mode coupling coefficient profile for a specific mode triad
            self.compute_mode_triad_profile()
        # - generate grid data
        elif compute_grid:
            # compute the mode coupling coefficient for the different mode triads and store the information in this solver object, as well as the stationary solutions
            # - CHECK if you want to display the progress bar for the computations
            if progress_bars:
                for _i in tqdm(
                    range(len(self._coupling_info)), desc='Grid computations'
                ):
                    self._call_grid_computations(info_index=_i)
            else:
                for _i in range(len(self._coupling_info)):
                    self._call_grid_computations(info_index=_i)
            # store the relevant mode statistics
            self.store_relevant_statistics()
        else:
            raise InlistError(
                msg="\nYou are not generating any data. Check if one of the inlist settings '[computation] - compute_grid_models' or '[computation] - compute_coefficient_profile' is set to True (if none of them are set to True, no data will be generated).",
                inlist_path=self._inlist_path,
            )

    def _call_grid_computations(self, info_index: int) -> None:
        """Performs the necessary computations for the grid.

        Parameters
        ----------
        info_index : int
            Index that selects the information for which the computations are performed (i.e. the combination number).
        """
        # compute the coupling for a single mode triad
        precheck = self.compute_coupling_one_triad(
            radial_order_combo_nr=info_index
        )
        # compute the stationary solutions, if necessary and store the theoretical amplitudes and observable fluctuations, when not using polytropic information
        if not self._polytrope_comp:
            self._stat_sol_info[info_index] = ThreeModeStationary(
                precheck,
                self._freq_handler,
                self._hyperbolicity_info[info_index],
                solver_object=self,
            )()
        else:
            self._stat_sol_info[info_index] = None

    @property
    def mode_freq_info(self) -> list[tuple]:
        """Return the mode frequency information.

        Returns
        -------
        list[tuple]
            Contains the mode frequency information.
        """
        return self._mode_freq_info

    @property
    def coupling_info(self) -> list[tuple]:
        """Return the coupling information.

        Returns
        -------
        list[tuple]
            Contains the coupling information.
        """
        return self._coupling_info

    @property
    def coupling_statistics(self) -> list[tuple]:
        """Return the coupling statistics.

        Returns
        -------
        list[tuple]
            Contains the coupling statistics.
        """
        return self._coupling_stat_info

    @property
    def coupler(self) -> QCCR | None:
        """Return the coupling coefficient-computing object.

        Returns
        -------
        QCCR
            The coupling coefficient-computing object.
        """
        return self._coupler

    @property
    def coupling_profile(self) -> np.ndarray | None:
        """Return the coupling coefficient profile.

        Returns
        -------
        np.ndarray
            The coupling coefficient profile.
        """
        return self._coupling_profile

    @property
    def stat_sol_info(self) -> list[dict[str, typing.Any] | None]:
        """Returns the stationary solution information.

        Returns
        -------
        list[dict]
            Contains the stationary solution information.
        """
        return self._stat_sol_info

    @property
    def linear_driving_rates(self) -> np.ndarray:
        """Returns the linear driving rates for the loaded mode triad.

        Returns
        -------
        np.ndarray
            Contains the linear driving/damping rates of the three loaded modes.
        """
        return self._driving_rates

    @property
    def corot_mode_omegas(self) -> np.ndarray:
        """Returns the co-rotating frame mode omegas for the loaded mode triad.

        Returns
        -------
        np.ndarray
            Contains the linear mode omegas in the co-rotating frame of the the loaded mode triad.
        """
        return self._freq_handler.corot_mode_omegas

    # load triad of nonradial gravity modes
    def load_nonradial_mode_triad(self, radial_order_combo_nr: int = 0) -> None:
        """Loads the necessary information of a triad of nonradial modes.

        Parameters
        ----------
        radial_order_combo_nr : int, optional
            The index number of the set of different combinations of radial orders; by default 0.
        """
        logger.debug('Reading mode info.')
        if self._compute_grid:
            _common_kwargs = {
                **self._info_dict_info_dirs,
                'infer_dir': False,
                'search_l': self._mode_info_object.mode_l,
                'search_m': self._mode_info_object.mode_m,
                'search_n': self._mode_info_object.combinations_radial_orders[
                    radial_order_combo_nr
                ],
            }
        elif self._compute_profile:
            # construct the radial order tuple
            self._rad_tup_profile = (self._cc_profile_selection_dict['parent_n'], self._cc_profile_selection_dict['daughter_1_n'], self._cc_profile_selection_dict['daughter_2_n'])
            # construct the common kwargs
            _common_kwargs = {
                **self._info_dict_info_dirs,
                'infer_dir': False,
                'search_l': self._cc_profile_selection_dict['mode_l'],
                'search_m': self._cc_profile_selection_dict['mode_m'],
                'search_n': self._rad_tup_profile,
            }
        # adjust the kwargs based on whether polytropes are used
        if self._polytrope_comp:
            # load adiabatic models
            _poly_mods = ModelLoader(
                base_dir=_common_kwargs['base_dir'],
                polytrope_model_output_dir=_common_kwargs['poly_struct_dir'],
                polytrope_oscillation_output_dir=_common_kwargs[
                    'poly_mode_dir'
                ],
                infer_dir=_common_kwargs['infer_dir'],
                search_l=_common_kwargs['search_l'],
                search_m=_common_kwargs['search_m'],
                search_n=_common_kwargs['search_n'],
                use_polytrope_data=True,
                polytrope_model_suffix=_common_kwargs['poly_struct_suffix'],
                polytrope_model_substrings=_common_kwargs[
                    'poly_struct_substrings'
                ],
                polytrope_oscillation_substrings=_common_kwargs[
                    'poly_mode_substrings'
                ],
            )  # initialize loader object
            _poly_mods.read_data_files(
                polytrope_oscillation=True,
                polytrope_model=True,
                g_modes=[True, True, True],
                polytrope_mass=self._polytrope_mass * M_sun.cgs.value,
                polytrope_radius=self._polytrope_radius * R_sun.cgs.value,
            )  # load modes TODO: generalize units!
            logger.debug('Mode info read.')
            # store the polytropic mode information in the object
            self._poly_models = _poly_mods.polytrope_oscillation_models
            # store polytropic structure information in the object
            self._poly_struct = _poly_mods.polytrope_model
            # set the other mode information attributes to None
            self._gyre_ad = None
            self._gyre_nad = None
            self._mesa = None
            logger.debug('Mode information stored.')
        else:
            # pop polytrope elements
            _common_kwargs.pop('poly_base_dir')
            _common_kwargs.pop('poly_struct_dir')
            _common_kwargs.pop('poly_struct_substrings')
            _common_kwargs.pop('poly_struct_suffix')
            _common_kwargs.pop('poly_mode_base_dir')
            _common_kwargs.pop('poly_mode_dir')
            _common_kwargs.pop('poly_mode_substrings')
            _common_kwargs.pop('poly_summary_substring')
            # load adiabatic models
            _mods_ad = ModelLoader(
                gyre_output_dir=self._ad_base_dir, **_common_kwargs
            )  # initialize loader object
            _mods_ad.read_data_files(
                gyre_detail=True, mesa_profile=True, g_modes=[True, True, True]
            )  # load modes
            # load non-adiabatic models
            _mods_nad = ModelLoader(
                gyre_output_dir=self._nad_base_dir, **_common_kwargs
            )
            _mods_nad.read_data_files(
                gyre_detail=True, mesa_profile=False, g_modes=[True, True, True]
            )
            logger.debug('Mode info read.')
            # store the GYRE information in the object
            self._gyre_ad = _mods_ad.gyre_details
            self._gyre_nad = _mods_nad.gyre_details
            # store MESA information in the object
            self._mesa = _mods_ad.mesa_profile
            # store the other mode information attributes to None
            self._poly_models = None
            self._poly_struct = None
            logger.debug('Mode information stored.')

    # initialize the mode coupling-computing object, and compute the mode coupling coefficient
    def compute_mode_coupling(
        self,
        l_list: list[int],
        m_list: list[int],
        radial_order_combo_nr: int = 0,
        precheck_satisfied: bool = True,
        two_mode_resonance: bool = False,
    ) -> None:
        """Computes the (quadratic) mode coupling coefficient for the selected non-radial mode triad.

        Parameters
        ----------
        l_list : list[int]
            The list of spherical degrees.
        m_list : list[int]
            The list of azimuthal orders.
        radial_order_combo_nr : int, optional
            The index number of the set of different combinations of radial orders; by default 0.
        precheck_satisfied : bool, optional
            Whether the (stationary) pre-check is satisfied for the mode triad under consideration; by default True.
        two_mode_resonance : bool, optional
            Whether the resonance is a two-mode (harmonic) resonance; by default False.
        """
        # initialize the mode coupling object
        if self._use_rot_formalism:
            # POLYTROPE
            if self._polytrope_comp:
                # issue warning about trying to use complex-valued eigenfunctions in the polytropic case
                if self._use_complex:
                    logger.warning(
                        'Tried to use complex-valued mode eigenfunctions in the polytropic case, for which mode eigenfunctions are real-valued. Check if this was your intention. Now proceeding with the computation of coupling coefficients using the real-valued adiabatic mode eigenfunctions.'
                    )
                if typing.TYPE_CHECKING:
                    assert isinstance(self._poly_struct, list)
                # initialize the coupling object
                self._coupler = QCCR(
                    *self._poly_models,
                    additional_profile_information=self._poly_struct[0],
                    freq_handler=self._freq_handler,
                    l_list=l_list,
                    m_list=m_list,
                    polytropic=True,
                    use_complex=False,
                    use_brunt_def=self._use_brunt_def,
                    use_symbolic_derivatives=self._use_symbolic_profiles,
                    nr_omp_threads=self._nr_omp_threads,
                    use_parallel=self._use_parallel,
                    numerical_integration_method=self._numerical_integration_method,
                    store_debug=self._get_debug_info,
                    conj_cc=self._terms_cc_conj,
                    analytic_polytrope=self._analytic_polytrope,
                    use_cheby_integration=self._use_cheby,
                    cheby_order_multiplier=self._cheby_order_mul,
                    cheby_blow_up_protection=self._cheby_blow_up,
                    cheby_blow_up_factor=self._cheby_blow_up_fac,
                )
            # NOT A POLYTROPE BUT ADIABATIC
            elif self._adiabatic:
                # issue warning about trying to use complex-valued eigenfunctions in the adiabatic case
                if self._use_complex:
                    logger.warning(
                        'Tried to use complex-valued mode eigenfunctions in the adiabatic case, for which mode eigenfunctions are real-valued. Check if this was your intention. Now proceeding with the computation of coupling coefficients using the real-valued adiabatic mode eigenfunctions.'
                    )
                if typing.TYPE_CHECKING:
                    assert isinstance(self._mesa, list)
                # initialize the coupling object, if needed
                if (
                    precheck_satisfied and self._stat_solv
                ) or not self._stat_solv:
                    self._coupler = QCCR(
                        *self._gyre_ad,
                        additional_profile_information=self._mesa[0],
                        freq_handler=self._freq_handler,
                        l_list=l_list,
                        m_list=m_list,
                        polytropic=False,
                        use_complex=False,
                        use_brunt_def=self._use_brunt_def,
                        use_symbolic_derivatives=self._use_symbolic_profiles,
                        nr_omp_threads=self._nr_omp_threads,
                        use_parallel=self._use_parallel,
                        numerical_integration_method=self._numerical_integration_method,
                        store_debug=self._get_debug_info,
                        conj_cc=self._terms_cc_conj,
                        use_cheby_integration=self._use_cheby,
                        cheby_order_multiplier=self._cheby_order_mul,
                        cheby_blow_up_protection=self._cheby_blow_up,
                        cheby_blow_up_factor=self._cheby_blow_up_fac,
                    )
                else:
                    self._coupler = None
            else:
                if typing.TYPE_CHECKING:
                    assert isinstance(self._mesa, list)
                # computations performed using either complex-valued eigenfunctions, or their real parts
                if (
                    precheck_satisfied and self._stat_solv
                ) or not self._stat_solv:
                    self._coupler = QCCR(
                        *self._gyre_nad,
                        additional_profile_information=self._mesa[0],
                        freq_handler=self._freq_handler,
                        l_list=l_list,
                        m_list=m_list,
                        polytropic=False,
                        use_complex=self._use_complex,
                        use_brunt_def=self._use_brunt_def,
                        use_symbolic_derivatives=self._use_symbolic_profiles,
                        nr_omp_threads=self._nr_omp_threads,
                        use_parallel=self._use_parallel,
                        numerical_integration_method=self._numerical_integration_method,
                        store_debug=self._get_debug_info,
                        conj_cc=self._terms_cc_conj,
                        use_cheby_integration=self._use_cheby,
                        cheby_order_multiplier=self._cheby_order_mul,
                        cheby_blow_up_protection=self._cheby_blow_up,
                        cheby_blow_up_factor=self._cheby_blow_up_fac,
                    )
                else:
                    self._coupler = None
        else:
            logger.error('Not yet implemented. Now exiting.')
            sys.exit()
        # computes the quadratic mode coupling coefficients if the pre-check is satisfied and we are computing stationary solutions, or when not computing stationary solutions --> finally, store that information
        self._store_mode_coupling_info(radial_order_combo_nr)
        # store mode frequency information
        self._store_mode_frequency_info(radial_order_combo_nr)
        # store whether the stationary solution fulfills the hyperbolicity condition
        self._store_hyperbolicity_info(
            radial_order_combo_nr, two_mode_resonance
        )

    # store whether the fixed point of the AEs fulfills the hyperbolicity condition
    def _store_hyperbolicity_info(
        self, radial_order_combo_nr: int, two_mode_resonance: bool
    ) -> None:
        """Stores whether the hyperbolicity condition is fulfilled for the specific mode triad.

        Notes
        -----
        Can only compute hyperbolicity info if a mode coupling coefficient is computed. If not computed, that value is set to zero, which might cause NaN issues when computing the entries in the Jacobian matrix.

        Parameters
        ----------
        radial_order_combo_nr : int
            The number used to denote a specific mode triad.
        two_mode_resonance : bool
            Whether the resonance is a two-mode (harmonic) resonance.
        """
        # If the pre-check is satisfied, check the hyperbolicity condition:
        if self._polytrope_comp:
            # ASSUME STABILITY BECAUSE WE CANNOT ASSESS THIS WITH POLYTROPES
            self._hyperbolicity_info[radial_order_combo_nr] = True
        elif self._coupler is not None:
            # determine whether the fixed point is hyperbolic for this mode triad
            # - compute the jacobian in the fixed point
            if two_mode_resonance:
                _my_jac = hyper_jac_three(
                    self._freq_handler.driving_rates,
                    self._freq_handler.corot_mode_omegas,
                    self._freq_handler.q_factor,
                    self._coupling_info[radial_order_combo_nr][1],
                )
            else:
                _my_jac = hyper_jac(
                    self._freq_handler.driving_rates,
                    self._freq_handler.corot_mode_omegas,
                    self._freq_handler.q_factor,
                    self._coupling_info[radial_order_combo_nr][1],
                )
            # - store the result of the hyperbolicity check
            self._hyperbolicity_info[radial_order_combo_nr] = check_hyper(
                _my_jac
            )
        else:
            # ALWAYS ASSUMED UNSTABLE OTHERWISE!
            self._hyperbolicity_info[radial_order_combo_nr] = False

    # store mode frequency information
    def _store_mode_frequency_info(self, radial_order_combo_nr: int) -> None:
        """Stores the mode frequency information for the specific triad of modes denoted by 'radial_order_combo_nr'.

        Parameters
        ----------
        radial_order_combo_nr : int
            The number used to denote a specific triad of modes.
        """
        # store the mode frequency information
        if self._polytrope_comp:
            self._mode_freq_info[radial_order_combo_nr] = (
                self._freq_handler.inert_mode_freqs,
                self._freq_handler.corot_mode_freqs,
                self._freq_handler.inert_mode_omegas,
                self._freq_handler.corot_mode_omegas,
                self._freq_handler.dimless_inert_freqs,
                self._freq_handler.dimless_corot_freqs,
                self._freq_handler.dimless_inert_omegas,
                self._freq_handler.dimless_corot_omegas,
                self._freq_handler.spin_factors,
                self._freq_handler.surf_rot_freq,
                self._freq_handler.requested_units,
            )
        else:
            self._mode_freq_info[radial_order_combo_nr] = (
                self._freq_handler.inert_mode_freqs,
                self._freq_handler.corot_mode_freqs,
                self._freq_handler.inert_mode_omegas,
                self._freq_handler.corot_mode_omegas,
                self._freq_handler.dimless_inert_freqs,
                self._freq_handler.dimless_corot_freqs,
                self._freq_handler.dimless_inert_omegas,
                self._freq_handler.dimless_corot_omegas,
                self._freq_handler.spin_factors,
                self._freq_handler.quality_factors,
                self._freq_handler.surf_rot_freq,
                self._freq_handler.requested_units,
                self._freq_handler.inert_mode_freqs_nad,
                self._freq_handler.corot_mode_freqs_nad,
                self._freq_handler.inert_mode_omegas_nad,
                self._freq_handler.corot_mode_omegas_nad,
                self._freq_handler.dimless_inert_freqs_nad,
                self._freq_handler.dimless_corot_freqs_nad,
                self._freq_handler.dimless_inert_omegas_nad,
                self._freq_handler.dimless_corot_omegas_nad,
            )

    # store mode coupling information
    def _store_mode_coupling_info(self, radial_order_combo_nr: int) -> None:
        """Stores the mode coupling information for the specific triad of modes denoted by 'radial_order_combo_nr'.

        Parameters
        ----------
        radial_order_combo_nr : int
            The number used to denote a specific triad of modes.
        """
        # compute the quadratic mode coupling coefficients if the pre-check is satisfied and we are computing stationary solutions, or when not computing stationary solutions
        if self._coupler is not None:
            # COMPUTE the coupling coefficient
            self._coupler.adiabatic_coupling_rot()
            # store the coupling coefficient + additional information
            # - (eta,nu,omega,adiabatic)
            cc = self._coupler.coupling_coefficient
            ncc = self._coupler.normed_coupling_coefficient
            elcc = self._coupler.eta_lee_coupling_coefficient
            if typing.TYPE_CHECKING:
                assert cc
                assert ncc
                assert elcc
            self._coupling_info[radial_order_combo_nr] = (
                cc,
                ncc,
                elcc,
                self._freq_handler.get_inert_corot_f().copy(),
                self._freq_handler.get_dimless_omegas().copy(),
                self._coupler.normalization_factors.copy(),
                self._adiabatic,
                self._freq_handler.get_detunings().copy(),
                self._coupler.coupling_terms_quadratic.copy(),
            )
        else:
            # store the zero-valued coupling coefficients + additional information - (eta,nu,omega,adiabatic)
            self._coupling_info[radial_order_combo_nr] = (
                0.0,
                0.0,
                0.0,
                self._freq_handler.get_inert_corot_f().copy(),
                self._freq_handler.get_dimless_omegas().copy(),
                np.zeros((3,), dtype=np.float64),
                self._adiabatic,
                self._freq_handler.get_detunings().copy(),
                np.zeros((4,), dtype=np.float64),
            )

    # compute disc integral factors
    def _compute_disc_integral_factors(self) -> None:
        """Computes the disc integral factors defined in Van Beeck et al. (2022) for the modes of the triad under consideration."""
        # initialize the disc integral factor arrays containing disc integral factors defined in Van Beeck et al. (2022)
        self._tk_disc_integrals = np.empty((3,), dtype=np.float64)
        self._ek_disc_integrals = np.empty_like(self._tk_disc_integrals)
        # obtain the mu values
        if typing.TYPE_CHECKING:
            assert self._coupler
        _muvals = self._coupler._mu_values
        # loop over the indices of the different modes in the triad
        for _i in range(1, 4):
            # obtain necessary information for the disc integral factor computation algorithm
            _my_hr = getattr(self._coupler, f'_hr_mode_{_i}')
            _my_der_hr = getattr(self._coupler, f'_theta_der_hr_mode_{_i}')
            _my_der2_hr = getattr(self._coupler, f'_theta_der2_hr_mode_{_i}')
            # compute and store the disc integrals
            (
                self._tk_disc_integrals[_i - 1],
                self._ek_disc_integrals[_i - 1],
            ) = di.get_integrals(
                _muvals,
                _my_hr,
                _my_der_hr,
                _my_der2_hr,
                self._ld_function,
                self._numerical_integration_method,
                self._nr_omp_threads,
                self._use_parallel,
            )

    # compute the mode coupling coefficient profile
    def compute_coupling_coefficient_profile(self) -> None:
        """Method that computes the mode coupling coefficient profile, after the normal coupling coefficients are computed, if possible."""
        # log message
        logger.debug('Now computing coupling coefficient profile!')
        # use the stored coupling object to compute the profile, if possible
        if self._coupler is None:
            logger.info(
                'No coupling object available. This means that the stationary solution is not stable!'
            )
            logger.debug(
                'No coupling coefficient profile was generated because the coupling coefficient was of a non-stationary solution!'
            )
        else:
            self._coupler.adiabatic_coupling_rot_profile()
            # store the coupling coefficient profile
            self._coupling_profile = (
                self._coupler.normed_coupling_coefficient_profile
            )
            # log message
            logger.debug('Coupling coefficient profile computed and stored.')

    # compute additional modal information for the nonradial mode triad
    def compute_additional_mode_info(
        self, radial_order_combo_nr: int = 0
    ) -> None:
        """Compute additional modal information for the loaded nonradial mode triad.

        Parameters
        ----------
        radial_order_combo_nr : int, optional
            The index number of the set of different combinations of radial orders; by default 0.
        """
        logger.debug('Computing additional mode information.')
        # # - QUALITY FACTORS
        # # compute the quality factors and their products
        # # (mode numbers for products: [1,2], [1,3], [2,3])
        # self._freq_handler.get_quality_factors(self._driving_rates)
        # compute the Lagrangian radiative luminosity perturbations
        # at the surface, and their moduli
        self._compute_lagrangian_surface_luminosity_perturbations(
            radial_order_combo_nr=radial_order_combo_nr
        )
        # compute the additional term needed to compute
        # the surface luminosity fluctuations
        self._compute_xir_term_norm(radial_order_combo_nr=radial_order_combo_nr)
        # compute the disc integral factors
        if self._coupler is not None:
            self._compute_disc_integral_factors()
        logger.debug('Additional mode information computed.')

    # individual methods that compute additional modal information
    def _compute_quality_factors_products(self) -> None:
        """Computes the quality factors of the individual modes and their products."""
        # compute and store the quality factors
        self._quality_factors = self._ang_mode_freqs / self._driving_rates
        # compute and store the quality factor products (mode numbers for products: [1,2], [1,3], [2,3])
        _q_prod = self._quality_factors * self._quality_factors[:, np.newaxis]
        self._quality_factor_products = _q_prod[np.tril_indices(3, k=-1)]

    def _get_angular_mode_frequencies_gyre(self) -> None:
        """Retrieves and stores the angular mode frequencies from either adiabatic or non-adiabatic GYRE output."""
        if typing.TYPE_CHECKING:
            assert self._gyre_ad
            assert self._gyre_nad
        self._ang_mode_freqs = np.array(
            [_x['freq'].real for _x in self._gyre_ad]
            if self._adiabatic
            else [_x['freq'].real for _x in self._gyre_nad]
        ).reshape(
            -1,
        )

    @staticmethod
    def _get_angular_mode_frequencies_gyre_parallel(
        gyre_ad: list[dict[str, typing.Any]],
        gyre_nad: list[dict[str, typing.Any]],
        adiabatic: bool,
    ):
        """Retrieves and stores the angular mode frequencies from either adiabatic or non-adiabatic GYRE output.

        Parameters
        ----------
        gyre_ad : list[dict[str, typing.Any]]
            Contains adiabatic GYRE information.
        gyre_nad : list[dict[str, typing.Any]]
            Contains non-adiabatic GYRE information.
        adiabatic : bool
            If True, store adiabatic data. If False, store non-adiabatic data.
        """
        return np.array(
            [_x['freq'].real for _x in gyre_ad]
            if adiabatic
            else [_x['freq'].real for _x in gyre_nad]
        ).reshape(
            -1,
        )

    def _get_linear_driving_rates_gyre(self) -> None:
        """Retrieves and stores the modal linear driving rates from the non-adiabatic GYRE output."""
        if typing.TYPE_CHECKING:
            assert self._gyre_nad
        self._driving_rates = np.array(
            [_x['freq'].imag for _x in self._gyre_nad]
        ).reshape(
            -1,
        )

    @staticmethod
    def _get_linear_driving_rates_gyre_parallel(
        gyre_nad: list[dict[str, typing.Any]]
    ) -> np.ndarray:
        """Retrieve the linear driving rates from GYRE simulations in parallel.

        Parameters
        ----------
        gyre_nad : list[dict[str, typing.Any]]
            Contains non-adiabatic GYRE information.

        Returns
        -------
        np.ndarray
            Contains the linear driving rates.
        """
        return np.array([_x['freq'].imag for _x in gyre_nad]).reshape(
            -1,
        )

    def _compute_lagrangian_surface_luminosity_perturbations(
        self, radial_order_combo_nr: int
    ):
        """Computes the Lagrangian luminosity perturbations at the surface due to the different modes, as computed by GYRE and renormalized using our normalization factor for the GYRE eigenfunctions.

        Parameters
        ----------
        radial_order_combo_nr : int
            The index number of the set of different combinations of radial orders.
        """
        if typing.TYPE_CHECKING:
            assert self._gyre_nad
        # - LAGRANGIAN PERTURBATION SURFACE RADIATIVE LUMINOSITY
        # retrieve the GYRE lagrangian perturbations of the radiative luminosities at the surface = Lag(L)/L
        _lag_rad_lum_gyre_surface = np.array(
            [re_im(_x['lag_L'][-1], 1.0)[0] for _x in self._gyre_nad],
            dtype=np.complex128,
        )
        # perform the normalization for Lag(L)/L
        _lag_rad_lum_norm = (
            _lag_rad_lum_gyre_surface
            * self._coupling_info[radial_order_combo_nr][-3]
        )
        # compute the radiative luminosity at the surface L_rad/L
        _rad_lum = np.array(
            [(_e['c_rad'][-1] * _e['x'][-1] ** (-3.0)) for _e in self._gyre_nad]
        )
        # compute the Lagrangian variations of surface luminosity Lag(L)/L,  and its modulus (renormalized using the normalizing factor for the eigenfunctions)
        self._lag_var_surf_lum = _lag_rad_lum_norm  # / _rad_lum

    def _compute_xir_term_norm(self, radial_order_combo_nr: int) -> None:
        """Computes the additional normalized term used to compute the luminosity fluctuations, as computed by GYRE and renormalized using our normalization factor for the GYRE eigenfunctions.

        Parameters
        ----------
        radial_order_combo_nr : int
            The index number of the set of different combinations of radial orders.
        """
        if typing.TYPE_CHECKING:
            assert self._gyre_ad
        # compute the additional normalized term used to convert the surface luminosity variation to the surface flux variation
        _xir_terms = np.array(
            [
                (re_im(_e['xi_r'][-1], 1.0)[0] / (_e['x'][-1]))
                for _e in self._gyre_ad
            ]
        )  # based on the adiabatic eigenfunction
        # renormalize
        self._xir_terms_norm = (
            _xir_terms * self._coupling_info[radial_order_combo_nr][-3]
        )

    # print mode information
    def print_mode_info(self, radial_order_combo_nr: int = 0):
        """Method that prints out the mode information.

        Parameters
        ----------
        radial_order_combo_nr : int, optional
            The index number of the set of different combinations of radial orders; by default 0.
        """
        if typing.TYPE_CHECKING:
            assert self._gyre_ad
        # store data in temporary variables
        _eta = self._coupling_info[radial_order_combo_nr][2]
        _abs_eta = np.absolute(_eta)
        _log_abs_eta = np.log10(_abs_eta)
        _log_lag_var = np.log10(self._lag_var_surf_lum)
        _q_factor_prod = np.sqrt(self._quality_factor_products[-1])
        _log_q_factor_prod = np.log10(_q_factor_prod)
        _dimless_omega_a_cor = self._coupling_info[radial_order_combo_nr][-4][0]
        _dimless_omega_a_in = self._coupling_info[radial_order_combo_nr][-4][1]
        # compute the equilibrium amplitudes --> TODO: update this!!!
        coefficient_factor = _abs_eta * _q_factor_prod
        coefficient_factor = 1.0 / coefficient_factor
        factor = 1.0 + (
            (self._ang_mode_freqs[0] - self._ang_mode_freqs[1:].sum())
            / (self._driving_rates[0] - self._driving_rates[1:].sum())
        ) ** (2.0)
        factor_parametric = 1.0 + (
            (self._ang_mode_freqs[0] - self._ang_mode_freqs[1:].sum())
            / (-self._driving_rates[1:].sum())
        )
        # generate string that will be logged as a statement
        _my_log_string = '\n' * 3 + '-----------------------------------\n'
        _my_log_string += f'eta_ABC = {_eta}\n' + f'|eta_ABC| = {_abs_eta}\n'
        _my_log_string += f'log(|eta_ABC|) = {_log_abs_eta}\n'
        _my_log_string += f'log(Delta L_rad / L_rad)_surf = {_log_lag_var}\n'
        _my_log_string += f'(Q_B Q_C)^(1/2) = {_q_factor_prod}\n'
        _my_log_string += f'log((Q_B Q_C)^(1/2)) = {_log_q_factor_prod}\n'
        _my_log_string += f'Eq_A = {coefficient_factor * np.sqrt(factor)}\n'
        _my_log_string += f'parametric_A = {coefficient_factor * np.sqrt(factor_parametric)}\n'
        _my_log_string += f'Angular mode frequencies = {self._ang_mode_freqs}\n'
        _my_log_string += f'Dimless omega_A (corot) = {_dimless_omega_a_cor}\n'
        _my_log_string += (
            f'Dimless omega_A (inertial) = {_dimless_omega_a_in}\n'
        )
        _my_log_string += f'Excitation_rates = {self._driving_rates}\n'
        _my_log_string += (
            f'Quality factors (Lee, 2012) = {self._quality_factors}\n'
        )
        _my_log_string += (
            f'Delta(gamma) (must be < 0) = {self._driving_rates.sum()}\n'
        )
        _my_log_string += f'Using Adiabatic eigenfunctions to compute coupling coefficients = {self._adiabatic}\n'
        _my_log_string += (
            f"Radial orders = {[_x['n_g'] for _x in self._gyre_ad]}\n"
        )
        _my_log_string += '\n-----------------------------------' + '\n\n'
        # log the string at the info level
        logger.info(_my_log_string)

    # load 1 specific mode triad and compute the necessary details and  coupling coefficients (NOT THE COUPLING COEFFICIENT PROFILE)
    def compute_coupling_one_triad(
        self, radial_order_combo_nr: int = 0
    ) -> bool:
        """Computes the coupling coefficient and extra information for the mode triad with combination number 'radial_order_combo_nr'.

        Parameters
        ----------
        radial_order_combo_nr : int, optional
            The index number of the set of different combinations of radial orders; by default 0.

        Returns
        -------
        precheck_satisfied : bool
            Whether the precheck is satisfied.
        """
        if typing.TYPE_CHECKING:
            assert self._poly_models
            assert self._gyre_ad
        # load the modal information
        self.load_nonradial_mode_triad(
            radial_order_combo_nr=radial_order_combo_nr
        )
        # store the frequency handler object
        # - initialize the frequency handler object
        if self._polytrope_comp:
            self._freq_handler = FH(*(self._poly_models))
            # no pre-check done because no non-adiabatic values loaded:
            precheck_satisfied = True
            # generate l and m lists
            _my_l_list = [pm['l'] for pm in self._poly_models]
            _my_m_list = [pm['m'] for pm in self._poly_models]
            # check if this is a two-mode resonance
            two_mode_resonance = self.check_two_mode_sum_resonance(
                self._poly_models
            )
        else:
            self._freq_handler = FH(
                *(self._gyre_ad if self._adiabatic else self._gyre_nad)
            )
            # - load the driving rates
            self._freq_handler.set_mode_nonadiabatic_info(*self._gyre_nad)
            # determine whether we are dealing with a two-mode harmonic sum frequency resonance or a distinct three-mode sum frequency resonance
            two_mode_resonance = self.check_two_mode_sum_resonance(
                self._gyre_ad
            )
            # perform pre-check for the modes
            # - stationarity imposed
            # - coupling coefficient selection rules +
            # - stable stationary solution rules imposed
            if self._stat_solv:
                # initialize check object
                self._check_obj = PreThreeQuad(
                    self._gyre_ad,
                    self._freq_handler,
                    triads=[(1, 2, 3)],
                    conjugation=None
                    if self._terms_cc_conj is None
                    else [self._terms_cc_conj],
                    two_mode_harmonic=two_mode_resonance,
                )
                # perform check
                precheck_satisfied = self._check_obj.check()
            # - no stationarity imposed:
            # - check coupling coefficient selection rules only
            else:
                # initialize check object
                self._check_obj = PreCheckQuadratic(
                    self._gyre_ad,
                    triads=[(1, 2, 3)],
                    conjugation=None
                    if self._terms_cc_conj is None
                    else [self._terms_cc_conj],
                )
                # perform check
                precheck_satisfied = self._check_obj.generic_check()
            if typing.TYPE_CHECKING:
                assert isinstance(precheck_satisfied, bool)
            # generate l and m lists
            _my_l_list = self._check_obj.l_list[0]
            _my_m_list = self._check_obj.m_list[0]
        # compute the mode coupling coefficient
        self.compute_mode_coupling(
            l_list=_my_l_list,
            m_list=_my_m_list,
            radial_order_combo_nr=radial_order_combo_nr,
            precheck_satisfied=precheck_satisfied,
            two_mode_resonance=two_mode_resonance,
        )
        # compute the additional mode information, when polytropes are not used (=non-adiabatic information)
        if not self._polytrope_comp:
            self.compute_additional_mode_info(
                radial_order_combo_nr=radial_order_combo_nr
            )
        # return whether the pre-check is satisfied
        return precheck_satisfied

    # check if we are dealing with a two-mode or three-mode sum frequency resonance
    @staticmethod
    def check_two_mode_sum_resonance(
        gyre_ad: list[dict[str, typing.Any]]
    ) -> bool:
        """Check if a resonance is a two-mode resonance, or a three-mode resonance based on radial orders.

        Parameters
        ----------
        gyre_ad : list[dict[str, typing.Any]]
            Contains adiabatic GYRE information.

        Returns
        -------
        bool
            True if we are dealing with a two-mode harmonic sum frequency resonance. False if we are dealing with a (distinct) three-mode sum frequency resonance.
        """
        # check if radial order, degree and azimuthal order are the same for mode 2 and 3
        _n = gyre_ad[1]['n_pg'] == gyre_ad[2]['n_pg']
        _m = gyre_ad[1]['m'] == gyre_ad[2]['m']
        _l = gyre_ad[1]['l'] == gyre_ad[2]['l']
        return _m & _n & _l

    # get masking information based on the TAR validity check
    def get_masking_info_tar_validity(
        self
    ) -> 'tuple[npt.NDArray[np.bool_], list[tuple[int, int]] | list[tuple[Literal[0], int]], Sequence[slice | None], Sequence[slice | None], bool]':
        """Retrieves masking information that helps compute coupling coefficient contributions from zones within the stellar model in which the TAR is not valid.

        Returns
        -------
        _type_
            _description_
        """
        if typing.TYPE_CHECKING:
            assert self._coupler
        # TODO: RIDDEN WITH MISTAKES, FIX!!!!!
        # CHECK if the quality factors fulfill the check that the modes are on the slow manifold (quality factors > 10)!
        slow_manifold_check = (
            np.abs(self._coupler._freq_handler._quality_factors) > 10.0
        ).all()
        # retrieve the necessary information to perform the TAR validity check
        brunt_rps = np.sqrt(self._coupler._brunt_N2_profile)  # RAD PER SEC
        _my_coriolis_freq_rps = (
            2.0 * self._coupler._freq_handler.surf_rot_angular_freq_rps
        )
        # get the Brunt conversion factor for individual mode frequency comparison
        _brunt_conv = self._coupler._freq_handler.angular_conv_factor_from_rps
        # generate the TAR validity mask
        tar_rotation_check = np.abs(0.1 * brunt_rps) > np.abs(
            _my_coriolis_freq_rps
        )
        tar_individual_freq_check = (
            np.abs(0.1 * _brunt_conv * brunt_rps)[:, np.newaxis]
            > np.abs(self._coupler._freq_handler._corot_mode_omegas)
        ).all(axis=1)
        tar_valid: 'npt.NDArray[np.bool_]' = (
            tar_rotation_check & tar_individual_freq_check
        )
        # retrieve the TAR validity change indices
        _mask_start_validity = (
            (tar_valid[:-1] < tar_valid[1:]).nonzero()[0] + 1
        ).tolist()
        _mask_end_validity = (
            (tar_valid[:-1] > tar_valid[1:]).nonzero()[0]
        ).tolist()
        # get separate masks for zones in which the TAR holds due to frequency hierarchy of the individual modes (in the co-rotating frame) and Coriolis frequency
        # - NO TAR VALID ZONES
        if not slow_manifold_check:
            raise PhysicsError(
                msg='The modes in the triad are not on the slow manifold (i.e. the absolute values of their quality factors = co-rotating angular mode frequencies / linear driving or damping rates are not all greater than 10.0). This triad cannot be described using amplitude equations!'
            )
        elif ((start_len := len(_mask_start_validity)) == 0) & (
            (end_len := len(_mask_end_validity)) == 0
        ):
            # find reason why there are no TAR validity zones and warn about this!
            if (
                (
                    my_individual_uni := np.unique(tar_individual_freq_check)
                ).shape[0]
                == 1
            ) & (
                (my_rot_uni := np.unique(tar_rotation_check)).shape[0] != 1
            ) and not my_individual_uni[0]:
                logger.warning(
                    'No zone found in which the TAR approximation holds according to frequency hierarchies. Specifically, the individual mode frequencies in the co-rotating frame are not much smaller than any local Brunt-Väisälä frequency!'
                )
            elif (
                (my_individual_uni.shape[0] != 1)
                and (my_rot_uni.shape[0] == 1)
                and not my_rot_uni[0]
            ):
                logger.warning(
                    'No zone found in which the TAR approximation holds according to frequency hierarchies. Specifically, the Coriolis frequency is not much smaller than any local Brunt-Väisälä frequency!'
                )
            elif (
                (my_individual_uni.shape[0] == 1)
                and (my_rot_uni.shape[0] == 1)
                and not my_rot_uni[0]
                and not my_individual_uni[0]
            ):
                logger.warning(
                    'No zone found in which the TAR approximation holds according to frequency hierarchies. Specifically, both the Coriolis frequency and the individual mode frequencies in the co-rotating frame are not much smaller than any local Brunt-Väisälä frequency!'
                )
            else:
                raise PhysicsError(
                    msg=f'No validity zones were found for the TAR approximation, and it is not entirely clear why this is the case. (checking individual mode frequency hierarchy yields positive values at indices: {np.where(tar_individual_freq_check)[0]}; checking coriolis frequency hierarchy yields positive values at indices: {np.where(tar_rotation_check)[0]}).'
                )
            # set the selection masks and indices
            validity_zone_masks = [np.s_[None]]
            non_validity_indices = [(0, len(brunt_rps) - 1)]
            non_validity_zone_masks = [np.s_[:]]
            fully_invalid = True
        elif start_len == end_len:
            # LAST VALIDITY/NON-VALIDITY ZONE AT SURFACE
            if (_s_i := _mask_start_validity[0]) > (
                _s_e := _mask_end_validity[0]
            ):
                # TAR-valid core and TAR-valid surface layer
                validity_zone_masks = (
                    [np.s_[: (_s_e + 1)]]
                    + [
                        np.s_[_s : (_e + 1)]
                        for _s, _e in zip(
                            _mask_start_validity[:-1], _mask_end_validity[1:]
                        )
                    ]
                    + [np.s_[(_mask_start_validity[-1]) :]]
                )
                non_validity_indices = [
                    (_e + 1, _s)
                    for _s, _e in zip(_mask_start_validity, _mask_end_validity)
                ]
                non_validity_zone_masks = [
                    np.s_[(_e + 1) : _s]
                    for _s, _e in zip(_mask_start_validity, _mask_end_validity)
                ]
                logger.warning(
                    'TAR-valid core and TAR-valid surface layer detected.'
                )
            else:
                # TAR-invalid core and TAR-invalid surface layer
                validity_zone_masks = [
                    np.s_[_s : (_e + 1)]
                    for _s, _e in zip(_mask_start_validity, _mask_end_validity)
                ]
                non_validity_indices: 'list[tuple[Literal[0], int]]' = (
                    [(0, _s_i)]
                    + [
                        ((_e + 1), _s)
                        for _s, _e in zip(
                            _mask_start_validity[1:], _mask_end_validity[:-1]
                        )
                    ]
                    + [(_mask_end_validity[-1] + 1, len(brunt_rps) - 1)]
                )
                non_validity_zone_masks = (
                    [np.s_[:_s_i]]
                    + [
                        np.s_[(_e + 1) : _s]
                        for _s, _e in zip(
                            _mask_start_validity[1:], _mask_end_validity[:-1]
                        )
                    ]
                    + [np.s_[(_mask_end_validity[-1] + 1) :]]
                )
                logger.warning(
                    'TAR-invalid core and TAR-invalid surface layer detected.'
                )
            fully_invalid = False
        elif start_len == (end_len + 1):
            # TAR-invalid core and TAR-valid surface layer
            validity_zone_masks = [
                np.s_[_s : (_e + 1)]
                for _s, _e in zip(_mask_start_validity[:-1], _mask_end_validity)
            ] + [np.s_[_mask_start_validity[-1] :]]
            non_validity_indices = [(0, _mask_start_validity[0])] + [
                (_e + 1, _s)
                for _s, _e in zip(_mask_start_validity[1:], _mask_end_validity)
            ]
            non_validity_zone_masks = [np.s_[: _mask_start_validity[0]]] + [
                np.s_[(_e + 1) : _s]
                for _s, _e in zip(_mask_start_validity[1:], _mask_end_validity)
            ]
            logger.warning(
                'TAR-invalid core and TAR-valid surface layer detected.'
            )
            fully_invalid = False
        elif (start_len + 1) == end_len:
            # TAR-valid core and TAR-invalid surface layer
            validity_zone_masks = [np.s_[: _mask_end_validity[0] + 1]] + [
                np.s_[_s : (_e + 1)]
                for _s, _e in zip(_mask_start_validity, _mask_end_validity[1:])
            ]
            non_validity_indices = [
                (_e + 1, _s)
                for _s, _e in zip(_mask_start_validity, _mask_end_validity[:-1])
            ] + [((_mask_end_validity[-1] + 1), len(brunt_rps) - 1)]
            non_validity_zone_masks = [
                np.s_[(_e + 1) : _s]
                for _s, _e in zip(_mask_start_validity, _mask_end_validity[:-1])
            ] + [np.s_[(_mask_end_validity[-1] + 1) :]]
            logger.warning(
                'TAR-valid core and TAR-invalid surface layer detected.'
            )
            fully_invalid = False
        else:
            raise PhysicsError(
                msg=f'This is a weird error because it means that the indices obtained from the mask-determining part of this method fails to find reasonable start and end indices: start indices of TAR-valid zones = {_mask_start_validity}, end indices of TAR-valid zones = {_mask_end_validity}. Please check your input!'
            )
        # return the necessary values
        return (
            tar_valid,
            non_validity_indices,
            validity_zone_masks,
            non_validity_zone_masks,
            fully_invalid,
        )

    # compute and show coefficient profile for 1 mode triad
    def compute_mode_triad_profile(
        self, view_term_by_term: bool = True, view_kernels: bool = True
    ) -> None:
        """Computes the mode coupling coefficient profile
        for 1 specific mode triad.

        Parameters
        ----------
        radial_order_combo_nr : int, optional
            The index number of the set of different combinations of
            radial orders; by default 0.
        view_kernels : bool, optional
            If True, view the radial and angular term contributions of each of the four individual terms to the coupling coefficient. If False, do not show term contribution; by default False.
        view_term_by_term : bool, optional
            If True, view radial kernels of each of the four individual terms of the coupling coefficient. If False, do not show these radial kernels; by default False.
        """
        if typing.TYPE_CHECKING:
            assert self._coupler
            assert self._coupling_profile        
        # compute the mode coupling coefficient based on the input radial orders
        _ = self.compute_coupling_one_triad()
        # after loading the necessary information (e.g. the CC), compute the mode coupling coefficient profile
        self.compute_coupling_coefficient_profile()

        # compute the masking information related to the validity of the TAR
        (
            _tar_valid,
            _convection_zone_selection_indices,
            _validity_zone_masks,
            _convection_zone_masks,
            _fully_invalid,
        ) = self.get_masking_info_tar_validity()

        # store part of the save name that is model-dependent
        _model_name_part = get_model_dependent_part_save_name(
            save_name=self._save_name,
            profile_selection_dict=self._cc_profile_selection_dict,
            rad_order_tuple=self._rad_tup_profile,
        )
        # get path string for figure output, with MESA-model-specific sub-directory
        _mesa_specific_substring = '/'.join(
            self._info_dict_info_dirs['mesa_profile_substrings'][:-1]
        )
        _rot_substring = f'rot_{self._rot_pct}'
        _my_figure_output_path = get_figure_output_base_path(
            mesa_specific_path=_mesa_specific_substring,
            rot_percent_string=_rot_substring,
        )
        # store the normalized radial coordinate in a dummy variable
        _norm_rad = self._coupler._x

        # make the simple CC profile figure
        _single_cc_fig = (
            single_integrated_radial_profile_figure_template_function(
                fractional_radius=_norm_rad,
                y_values=self._coupling_profile,
                y_axis_label=r'$\eta_1\,(r)$',
                my_brunt_profile=self._coupler._brunt_N2_profile,
            )
        )

        save_fig(
            figure_path=_my_figure_output_path,
            figure_base_name='cc_profile_single',
            figure_subdir='coupling_coefficient',
            model_name_part=_model_name_part,
            fig=_single_cc_fig,
            save_as_pdf=True,
        )

        # make the coupling coefficient profile figure
        (
            _my_cc,
            _my_adjusted_cc,
            _cc_fig,
            _cont_arr_masked,
        ) = double_masking_integrated_radial_profile_figure_template_function(
            radial_coordinate=_norm_rad,
            y_ax_data=self._coupling_profile,
            y_ax_label=[r'$\eta_1\,(r)$', r'$\eta_{1,\,{\rm adj}}\,(r)$'],
            convection_zone_selection_indices=_convection_zone_selection_indices,
            fully_invalid=_fully_invalid,
            my_brunt=self._coupler._brunt_N2_profile,
            convection_zone_masks=_convection_zone_masks,
            validity_zone_masks=_validity_zone_masks,
            tar_validity_mask=_tar_valid,
        )

        # save figure
        save_fig(
            figure_path=_my_figure_output_path,
            figure_base_name='cc_profile',
            figure_subdir='coupling_coefficient',
            model_name_part=_model_name_part,
            fig=_cc_fig,
        )
        print(self._hyperbolicity_info)
        # get stationary solutions
        _my_stat_sol_info = ThreeModeStationary(
            precheck_satisfied=True,
            freq_handler=self._freq_handler,
            hyperbolic=self._hyperbolicity_info[0],
            solver_object=self,
        )()
        
        print(_my_cc, type(_my_cc))
        print(_my_adjusted_cc, type(_my_adjusted_cc))

        # compute the adjusted quantities, print the results, and get the adjustment ratio
        _coupling_coefficient_adjust_ratio = (
            print_normed_cc_and_get_adjust_ratio(
                cc_val=_my_cc, adjusted_cc_val=_my_adjusted_cc
            )
        )
        # display the equilibrium and threshold amplitudes and get the theoretical stationary amplitudes
        _stat_amps = print_theoretical_amplitudes(
            stat_sol_info=_my_stat_sol_info,
            adjust_ratio=_coupling_coefficient_adjust_ratio,
        )
        # display the mode surface luminosity fluctuations and the parent threshold surface luminosities
        print_surface_luminosities(
            stat_sol_info=_my_stat_sol_info,
            adjust_ratio=_coupling_coefficient_adjust_ratio,
            stat_amps=_stat_amps,
        )
        # display the individual contributions due to TAR non-validity regions
        display_contributions(contribution_array_masked=_cont_arr_masked)
        # CHECK HYPERBOLICITY WITH THE CHANGED COUPLING COEFFICIENT
        print_check_hyperbolic(
            solving_object=self,
            freq_handler=self._freq_handler,
            adjusted_cc=_my_adjusted_cc,
        )

        # if requested, provide an overview of the different terms
        if view_term_by_term:
            if typing.TYPE_CHECKING:
                assert self._coupler.normed_contributions_coupling_coefficient_profile
            # retrieve term profiles
            (
                cc_term_1,
                cc_term_2,
                cc_term_3,
                cc_term_4,
            ) = self._coupler.normed_contributions_coupling_coefficient_profile
            # compute and display the term profiles
            with plt.style.context(('tableau-colorblind10',)):
                (
                    _my_cc_contributions,
                    _my_adjusted_cc_contributions,
                    _cc_contribution_fig,
                    _cont_arr_masked,
                ) = double_masking_integrated_radial_profile_figure_template_function(
                    radial_coordinate=_norm_rad,
                    y_ax_data=[cc_term_1, cc_term_2, cc_term_3, cc_term_4],
                    y_ax_label=[
                        r'$\eta_1^{(i)}(r)$',
                        r'$\eta_{1,\,{\rm adj}}^{(i)}(r)$',
                    ],
                    convection_zone_selection_indices=_convection_zone_selection_indices,
                    fully_invalid=_fully_invalid,
                    convection_zone_masks=_convection_zone_masks,
                    validity_zone_masks=_validity_zone_masks,
                    tar_validity_mask=_tar_valid,
                    use_legend=True,
                    y_legend_entries=[
                        r'$\eta_{1(,\,{\rm adj})}^{(1)}$',
                        r'$\eta_{1(,\,{\rm adj})}^{(2)}$',
                        r'$\eta_{1(,\,{\rm adj})}^{(3)}$',
                        r'$\eta_{1(,\,{\rm adj})}^{(4)}$',
                    ],
                )

                # save figure
                save_fig(
                    figure_path=_my_figure_output_path,
                    figure_base_name='cc_contributions_profile',
                    figure_subdir='coupling_coefficient_contributions',
                    model_name_part=_model_name_part,
                    fig=_cc_contribution_fig,
                )

                if typing.TYPE_CHECKING:
                    assert isinstance(_my_cc_contributions, list)
                    assert isinstance(_my_adjusted_cc_contributions, list)
                    assert isinstance(_cont_arr_masked, list)

                # print information on the term contributions
                print_cc_contribution_info(
                    cc_contribution=_my_cc_contributions[0],
                    adjusted_cc_contribution=_my_adjusted_cc_contributions[0],
                    masked_contribution=_cont_arr_masked[0],
                    contribution_identifier_string='first',
                )
                print_cc_contribution_info(
                    cc_contribution=_my_cc_contributions[1],
                    adjusted_cc_contribution=_my_adjusted_cc_contributions[1],
                    masked_contribution=_cont_arr_masked[1],
                    contribution_identifier_string='second',
                )
                print_cc_contribution_info(
                    cc_contribution=_my_cc_contributions[2],
                    adjusted_cc_contribution=_my_adjusted_cc_contributions[2],
                    masked_contribution=_cont_arr_masked[2],
                    contribution_identifier_string='third',
                )
                print_cc_contribution_info(
                    cc_contribution=_my_cc_contributions[3],
                    adjusted_cc_contribution=_my_adjusted_cc_contributions[3],
                    masked_contribution=_cont_arr_masked[3],
                    contribution_identifier_string='fourth',
                )
                # add some space in between prints
                print('\n\n')

        # if requested, generate plots that display the radial and angular kernels
        if view_kernels:
            # define paths used to save figures
            # - radial kernels
            _rad_ker_path = _my_figure_output_path / 'radial_kernels'
            _rad_ker_path.mkdir(parents=True, exist_ok=True)
            # - angular kernels
            _ang_ker_path = _my_figure_output_path / 'angular_kernels'
            _ang_ker_path.mkdir(parents=True, exist_ok=True)

            # load the different terms that will make up the angular kernels into a custom object
            # -- CC CONTRIBUTION 1
            cc_contribution_1_data_ang = AngularKernelData(
                coupling_object=self._coupler
            )
            cc_contribution_1_dict_ang = cc_contribution_1_data_ang(
                contribution_nr=1
            )
            # -- CC CONTRIBUTION 2
            cc_contribution_2_data_ang = AngularKernelData(
                coupling_object=self._coupler
            )
            cc_contribution_2_dict_ang = cc_contribution_2_data_ang(
                contribution_nr=2
            )
            # -- CC CONTRIBUTION 3
            cc_contribution_3_data_ang = AngularKernelData(
                coupling_object=self._coupler
            )
            cc_contribution_3_dict_ang = cc_contribution_3_data_ang(
                contribution_nr=3
            )
            # -- CC CONTRIBUTION 4
            cc_contribution_4_data_ang = AngularKernelData(
                coupling_object=self._coupler
            )
            cc_contribution_4_dict_ang = cc_contribution_4_data_ang(
                contribution_nr=4
            )

            if typing.TYPE_CHECKING:
                assert cc_contribution_1_dict_ang
                assert cc_contribution_2_dict_ang
                assert cc_contribution_3_dict_ang
                assert cc_contribution_4_dict_ang

            # plot kernels for cc contribution 1
            with plt.style.context(('tableau-colorblind10',)):
                create_angular_contribution_figures_for_specific_contribution(
                    ylabel_number=1,
                    contribution_data=cc_contribution_1_data_ang,
                    contribution_dictionary=cc_contribution_1_dict_ang,
                    nr_plot_cols=[2, 2, 2, 2, 2],
                    angular_kernel_path=_ang_ker_path,
                    model_name_part=_model_name_part,
                    single_entry=False,
                )

            # plot kernels for cc contribution 2
            with plt.style.context(('tableau-colorblind10',)):
                create_angular_contribution_figures_for_specific_contribution(
                    ylabel_number=2,
                    contribution_data=cc_contribution_2_data_ang,
                    contribution_dictionary=cc_contribution_2_dict_ang,
                    nr_plot_cols=[1],
                    angular_kernel_path=_ang_ker_path,
                    model_name_part=_model_name_part,
                    single_entry=True,
                )

            # plot kernels for cc contribution 3
            with plt.style.context(('tableau-colorblind10',)):
                create_angular_contribution_figures_for_specific_contribution(
                    ylabel_number=3,
                    contribution_data=cc_contribution_3_data_ang,
                    contribution_dictionary=cc_contribution_3_dict_ang,
                    nr_plot_cols=[3, 3, 3, 3, 3, 3, 3, 3],
                    angular_kernel_path=_ang_ker_path,
                    model_name_part=_model_name_part,
                    single_entry=False,
                )

            # plot kernels for cc contribution 4
            with plt.style.context(('tableau-colorblind10',)):
                create_angular_contribution_figures_for_specific_contribution(
                    ylabel_number=4,
                    contribution_data=cc_contribution_4_data_ang,
                    contribution_dictionary=cc_contribution_4_dict_ang,
                    nr_plot_cols=[3],
                    angular_kernel_path=_ang_ker_path,
                    model_name_part=_model_name_part,
                    single_entry=False,
                )

            # load the different terms that will make up the radial kernels into a custom object
            # -- CC CONTRIBUTION 1
            cc_contribution_1_data = RadialKernelData(
                coupling_object=self._coupler
            )
            cc_contribution_1_dict = cc_contribution_1_data(contribution_nr=1)
            # -- CC CONTRIBUTION 2
            cc_contribution_2_data = RadialKernelData(
                coupling_object=self._coupler
            )
            cc_contribution_2_dict = cc_contribution_2_data(contribution_nr=2)
            # -- CC CONTRIBUTION 3
            cc_contribution_3_data = RadialKernelData(
                coupling_object=self._coupler
            )
            cc_contribution_3_dict = cc_contribution_3_data(contribution_nr=3)
            # -- CC CONTRIBUTION 4
            cc_contribution_4_data = RadialKernelData(
                coupling_object=self._coupler
            )
            cc_contribution_4_dict = cc_contribution_4_data(contribution_nr=4)

            # define a string used to denote an energy scaling factor part of a radial kernel
            energy_scale_string = r'$\,\left(\dfrac{{r^3}}{{\epsilon}}\right)$'

            if typing.TYPE_CHECKING:
                assert cc_contribution_1_dict
                assert cc_contribution_2_dict
                assert cc_contribution_3_dict
                assert cc_contribution_4_dict

            # plot kernels for cc contribution 1
            with plt.style.context(('tableau-colorblind10',)):
                create_radial_contribution_figures_for_specific_contribution(
                    ylabel_number=1,
                    contribution_data=cc_contribution_1_data,
                    contribution_dictionary=cc_contribution_1_dict,
                    nr_plot_cols=[3, 3],
                    radial_kernel_path=_rad_ker_path,
                    model_name_part=_model_name_part,
                    tar_valid=_tar_valid,
                    energy_scale_string=energy_scale_string,
                    single_entry=False,
                )

            # plot kernels for cc contribution 2
            with plt.style.context(('tableau-colorblind10',)):
                create_radial_contribution_figures_for_specific_contribution(
                    ylabel_number=2,
                    contribution_data=cc_contribution_2_data,
                    contribution_dictionary=cc_contribution_2_dict,
                    nr_plot_cols=[1],
                    radial_kernel_path=_rad_ker_path,
                    model_name_part=_model_name_part,
                    tar_valid=_tar_valid,
                    energy_scale_string=energy_scale_string,
                    single_entry=True,
                )

            # plot kernels for cc contribution 3
            with plt.style.context(('tableau-colorblind10',)):
                create_radial_contribution_figures_for_specific_contribution(
                    ylabel_number=3,
                    contribution_data=cc_contribution_3_data,
                    contribution_dictionary=cc_contribution_3_dict,
                    nr_plot_cols=[3, 3, 3, 3],
                    radial_kernel_path=_rad_ker_path,
                    model_name_part=_model_name_part,
                    tar_valid=_tar_valid,
                    energy_scale_string=energy_scale_string,
                    single_entry=False,
                )

            # plot kernels for cc contribution 4
            with plt.style.context(('tableau-colorblind10',)):
                create_radial_contribution_figures_for_specific_contribution(
                    ylabel_number=4,
                    contribution_data=cc_contribution_4_data,
                    contribution_dictionary=cc_contribution_4_dict,
                    nr_plot_cols=[3],
                    radial_kernel_path=_rad_ker_path,
                    model_name_part=_model_name_part,
                    tar_valid=_tar_valid,
                    energy_scale_string=energy_scale_string,
                    single_entry=False,
                )

        plt.show()
        import sys

        sys.exit()

    # store the relevant information of the specific mode triad for the mode coupling coefficient statistics plot
    def store_relevant_statistics(self) -> None:
        """Method that stores the relevant statistics for the mode triad that is currently loaded, so that a statistics plot can be generated."""
        # retrieve and store the relevant mode statistics
        for _j, (_avg_rad, _wu) in enumerate(
            zip(
                self._mode_info_object.average_radial_orders,
                self._mode_info_object.wu_difference_statistics,
            )
        ):
            # store the mode statistics
            self._coupling_stat_info[_j] = (_avg_rad, _wu)
