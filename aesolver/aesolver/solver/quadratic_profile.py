'''Python module containing subclass used to perform quadratic grid computations.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
'''
# import statements
import logging
import sys
import typing
import numpy as np
import matplotlib.pyplot as plt
from astropy.constants import M_sun, R_sun  # type: ignore (although the type checker does not find it, R_sun is part of astropy.constants)
from pathlib import Path
from tqdm import tqdm

# relative imports
# - used to load the superclass
from .solver_base_classes import QuadraticAESolver, PhysicsError

# - computes stationary solutions for A = B + C
from ..stationary_solving import ThreeModeStationary

# - used to plot radial kernel profiles
from .profile_helper_classes import RadialKernelData, AngularKernelData

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
    from typing import Any
    import numpy.typing as npt
    from typing import Literal
    from collections.abc import Sequence
    
    
logger = logging.getLogger(__name__)
    
    
class QuadraticAEProfileSolver(QuadraticAESolver):
    """The class that handles profile coupling coefficient profile computations.

    Parameters
    ----------
    view_kernels : bool, optional
        If True, view the radial and angular term contributions of each of the four individual terms to the coupling coefficient. If False, do not show term contribution; by default False.
    view_term_by_term : bool, optional
        If True, view radial kernels of each of the four individual terms of the coupling coefficient. If False, do not show these radial kernels; by default False.
    """
    
    # attribute type declarations
    
    def __init__(self, sys_args: list, cc_profile_selection_dict: dict[str, int], called_from_pytest: bool = False, pytest_path: Path | None = None, rot_pct: int = 20, use_complex: bool = False, adiabatic: bool = True, use_rotating_formalism: bool = True, base_dir: str | None = None, alternative_directory_path: str | None = None, alternative_file_name: str | None = None, use_symbolic_profiles: bool = True, use_brunt_def: bool = True, nr_omp_threads: int = 4, use_parallel: bool = False, ld_function: str = 'eddington', numerical_integration_method: str = 'trapz', use_cheby_integration: bool = False, cheby_order_multiplier: int = 4, cheby_blow_up_protection: bool = False, cheby_blow_up_factor: float = 1000, stationary_solving: bool = True, ad_path: str | None = None, nad_path: str | None = None, inlist_path: str | None = None, toml_use: bool = False, get_debug_info: bool = False, terms_cc_conj: tuple[bool, bool, bool] | bool | None = None, polytrope_comp: bool = False, analytic_polytrope: bool = False, polytrope_mass: float = 3, polytrope_radius: float = 4.5, save_name: str = '', view_term_by_term: bool=True, view_kernels: bool=True, progress_bars: bool=False) -> None:
        # initialize the super-class
        super().__init__(sys_args, called_from_pytest, pytest_path, rot_pct, use_complex, adiabatic, use_rotating_formalism, base_dir, alternative_directory_path, alternative_file_name, use_symbolic_profiles, use_brunt_def, nr_omp_threads, use_parallel, ld_function, numerical_integration_method, use_cheby_integration, cheby_order_multiplier, cheby_blow_up_protection, cheby_blow_up_factor, stationary_solving, ad_path, nad_path, inlist_path, toml_use, get_debug_info, terms_cc_conj, polytrope_comp, analytic_polytrope, polytrope_mass, polytrope_radius, progress_bars)
        # store the coefficient profile selection data dict
        self._cc_profile_selection_dict = cc_profile_selection_dict
                # initialize the lists that will hold the necessary information
        # - mode coupling information
        self._coupling_info = [tuple()]
        # - mode statistics information
        self._coupling_stat_info = [tuple()]
        # - stationary solution information
        self._stat_sol_info = [dict()]
        # - mode frequency information
        self._mode_freq_info = [tuple()]
        # - hyperbolicity (of fixed point) information
        self._hyperbolicity_info = [False]
        # store the file save name
        self._save_name = save_name
        # store viewing options
        self._view_term_by_term = view_term_by_term
        self._view_kernels = view_kernels
        
    # make the class a callable so that the requested data can be computed
    def __call__(
        self,
    ) -> None:
        # compute and plot the mode coupling coefficient profile, as well as other relevant profiles, for a specific mode triad
        self.compute_mode_triad_profile()
        
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
            self._coupler.adiabatic_coupling_rot_profile(progress_bar=self._display_progress_bars)
            # store the coupling coefficient profile
            self._coupling_profile = (
                self._coupler.normed_coupling_coefficient_profile
            )
            # log message
            logger.debug('Coupling coefficient profile computed and stored.')

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
    def compute_mode_triad_profile(self) -> None:
        """Computes the mode coupling coefficient profile for 1 specific mode triad."""
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
        if self._view_term_by_term:
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
        if self._view_kernels:
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
        
    # generate loading kwargs
    def _generate_loading_kwargs(self, radial_order_combo_nr: int=0) -> 'dict[str, Any]':
        # construct the radial order tuple
        self._rad_tup_profile = (self._cc_profile_selection_dict['parent_n'], self._cc_profile_selection_dict['daughter_1_n'], self._cc_profile_selection_dict['daughter_2_n'])
        # construct and return the common kwargs
        return {
            **self._info_dict_info_dirs,
            'infer_dir': False,
            'search_l': self._cc_profile_selection_dict['mode_l'],
            'search_m': self._cc_profile_selection_dict['mode_m'],
            'search_n': self._rad_tup_profile,
            }
