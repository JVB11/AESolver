"""Python package containing the class that will handle the plotting of the solutions of the quadratic amplitude equations.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import h5py as h5
import logging
import numpy as np
from multimethod import multimethod
from pathlib import Path

# import custom local site modules
# from aesolver.frequency_handling.enumeration_files import EnumGYREFreqConv

# import custom modules
from aesolver import generate_load_file_path_list

# submodule imports
from .resonance_sharpness import ResonanceSharpnessFigure
from .overview_plot import OverviewFigure
from .data_files import HDF5MappingsArrays, HDF5MappingsAttributes
from .library_functions import (
    isolated_non_isolated_stable_valid_masking,
    theoretical_plotting_masks_overview,
)
from .utility_functions import hz_to_cpd, hz_to_muhz, compute_single_amplitude_ratio

# type checking imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # TODO: use this for more generic unit conversions!
    # EnumGYREFreqConv
    # generic typing imports
    from collections.abc import Sequence
    from typing import TypedDict, NotRequired

    
    # typed dictionaries
    class SaveDataDict(TypedDict):
        save_dir: str | Path
        extra_specifier: str
        save_formats: list[str]


    class SpecificSaveInfoDict(TypedDict):
        sharpness_save_name: NotRequired[str]
        sharpness_subdir: NotRequired[str | Path]
        overview_save_name: NotRequired[str]
        overview_subdir: NotRequired[str | Path]


    from .data_files.hdf5_data_path_mapper import AttrMap, ArrMap


# only show warning and higher level logs for matplotlib package
logging.getLogger('matplotlib').setLevel(level=logging.WARNING)
# set up regular logger
logger = logging.getLogger(__name__)


# class that handles the plotting of the solutions of the amplitude equations
class QuadraticAEPlotter:
    """Class that handles the plotting of the solutions of the
    quadratic amplitude equations for gravity-mode pulsating stars.

    TODO: update this!

    Parameters
    ----------
    solver_object : QuadraticAESolver
        The object in which computations were performed to generate the plots.
    plotting_dict : dict
        The dictionary containing input arguments for the plotting.
    plot_cyclic_detunings : bool, optional
        If True, generate cyclic detunings instead of the angular frequency. If False, do not.
        Default: False.
    """

    # attribute type declarations TODO: update
    _hdf5_path: 'str | Sequence[str | list[str]]'
    _plot_overview: bool
    _verbose: bool
    _plot_coefficient_profile: bool
    _plot_resonance_sharpness: bool
    _fig_size: list[float] | tuple[float, float]
    _fig_dpi: int
    _ax_label_size: int
    _tick_label_size: int
    _tick_size: int
    _rad_nr: int
    _show_plots: bool
    _cmap: str
    _tick_label_pad: int
    _freq_unit: str
    # loaded data attributes
    _load_mappings_arrays: 'ArrMap'
    _load_mappings_attrs: 'AttrMap'
    # save attributes
    _generic_save_data_dictionary: 'SaveDataDict'
    _specific_save_data_dictionary: 'SpecificSaveInfoDict'

    # initialization method
    def __init__(
        self,
        load_information,
        plotting_dict,
        save_information,
        requested_frequency_unit='cpd',
    ) -> None:
        # store the path to the stored HDF5 file
        self._hdf5_path = generate_load_file_path_list(
            load_dict=load_information
        )
        # unpack and store the plotting dictionary's values
        self._unpack_plotting_dict(plot_dict=plotting_dict)
        # store the requested frequency unit
        self._freq_unit = requested_frequency_unit
        # use the specific unpacked plotting directives to decide which data to load and load it
        self._load_data()
        # store the basic save path, and additional information to generate file paths for saving generated figures/plots
        self._store_generic_save_information(save_dict=save_information)
        # store specific save information based on passed arguments
        self._store_specific_save_info(save_dict=save_information)

    def _store_generic_save_information(self, save_dict) -> None:
        # generate the basic information necessary to save plots
        # - path to base save directory
        _base_save_dir_path = Path(save_dict['save_base_dir'])
        # check if it exists, otherwise generate it
        _base_save_dir_path.mkdir(parents=True, exist_ok=True)
        # - path to save directory
        _save_dir_path = _base_save_dir_path / save_dict['save_dir']
        # check if it exists, otherwise generate it
        _save_dir_path.mkdir(parents=True, exist_ok=True)
        # store the generic save data dictionary
        self._generic_save_data_dictionary = {
            'save_dir': _save_dir_path,
            'extra_specifier': save_dict['extra_specifier_save'],
            'save_formats': save_dict['save_formats'],
        }

    def _store_specific_save_info(self, save_dict) -> None:
        # store the specific save dictionary
        self._specific_save_data_dictionary = {}
        # check if you need resonance sharpness save info
        if self._plot_resonance_sharpness:
            # add the file name
            self._specific_save_data_dictionary[
                'sharpness_save_name'
            ] = save_dict['sharpness_save_name']
            # add the subdirectory
            self._specific_save_data_dictionary['sharpness_subdir'] = save_dict[
                'sharpness_subdir'
            ]
        # check if you need overview plot save info
        if self._plot_overview:
            # add the file name
            self._specific_save_data_dictionary[
                'overview_save_name'
            ] = save_dict['overview_save_name']
            # add the subdirectory
            self._specific_save_data_dictionary['overview_subdir'] = save_dict[
                'overview_subdir'
            ]
        # check if you need coupling coefficient profile save info
        if self._plot_coefficient_profile:
            # TODO: implement this
            pass

    # loads the plot data during initialization
    def _load_data(self) -> None:
        if self._show_plots:
            logger.debug('You opted to view the generated plots.')
        else:
            logger.debug('You opted to not view any plots.')
        # get the load mappings (for HDF5 output)
        self._load_mappings_arrays = HDF5MappingsArrays().get_dict()
        # get the attr mappings (for HDF5 output)
        self._load_mappings_attrs = HDF5MappingsAttributes().get_dict()
        # load data agnostic to whether you want to plot a specific plot
        if self._plot_overview or self._plot_resonance_sharpness:
            self._load_sharpness_overview_data(self._hdf5_path)
            self._load_attributes()
            self._compute_frequency_data()
            self._compute_threshold_surface_luminosity_variations()
            self._compute_driv_damp_info()
            self._compute_amp_ratios()
            self._compute_cc_abs()
            # generate plotting masks
        # check if you need to load resonance sharpness info
        if self._plot_resonance_sharpness:
            logger.debug(
                'Now extracting data to make a resonance sharpness plot.'
            )
            # load the basic data for this plot
            self._store_resonance_sharpness_factors()
            # plotting masks for loaded data
            self._generate_plotting_masks_sharpness()
            # generate figure data
            self._generate_resonance_sharpness_figure_data()
            logger.debug('Data for resonance sharpness plot extracted.')
        if self._plot_overview:
            logger.debug('Now extracting data to make the overview plot.')
            # get the harmonic mask
            self._get_harmonic_mask()
            # compute frequency shifts
            self._compute_freq_shifts()
            # compute minimal inertial frequencies for mirroring
            self._compute_minimal_inertial_frequencies()
            # perform frequency mirroring
            self._perform_frequency_mirroring()
            # generate the plotting masks
            self._get_overview_plotting_masks()
            logger.debug('Data for overview plot extracted.')
        if self._plot_coefficient_profile:
            logger.debug(
                'Now extracting data to make the coupling coefficient profile plot.'
            )
            # TODO: implement this
            logger.debug(
                'Data for coupling coefficient profile plot extracted.'
            )

    # load data for the resonance sharpness and overview plots - single string
    @multimethod
    def _load_sharpness_overview_data(self, my_file_name_or_names: str) -> None:
        # load the file data and store it in the data dictionary
        self._list_data_dict = [
            self._load_specific_file_data(file_path=my_file_name_or_names)
        ]

    @multimethod
    def _load_sharpness_overview_data(
        self, my_file_name_or_names: list[str]
    ) -> None:
        # load the data for each of the individual files
        loaded_data_files = [
            self._load_specific_file_data(file_path=_linked_name)
            for _linked_name in my_file_name_or_names
        ]
        # load the file data and store it in the data dictionary
        self._list_data_dict = [
            {
                _k: np.concatenate([_d[_k] for _d in loaded_data_files], axis=0)
                for _k in self._load_mappings_arrays
            }
        ]

    @multimethod
    def _load_sharpness_overview_data(
        self, my_file_name_or_names: list[list[str]]
    ) -> None:
        # load the data for each of the individual files
        self._list_data_dict = []
        for _linked_name_or_list in my_file_name_or_names:
            if isinstance(_linked_name_or_list, str):
                self._list_data_dict.append(
                    self._load_specific_file_data(
                        file_path=_linked_name_or_list
                    )
                )
            else:
                files_single_object = []
                for _fn in _linked_name_or_list:
                    files_single_object.append(
                        self._load_specific_file_data(file_path=_fn)
                    )
                self._list_data_dict.append(
                    {
                        _k: np.concatenate(
                            [_d[_k] for _d in files_single_object], axis=0
                        )
                        for _k in self._load_mappings_arrays
                    }
                )

    def _load_specific_file_data(self, file_path: 'str | Path') -> None:
        # load data from a specific file
        with h5.File(file_path, 'r') as my_h5:
            my_data_dict = {
                _d_i: my_h5[_d_i_path][()]
                for _d_i, (_d_i_path, _) in self._load_mappings_arrays.items()
            }
        # return the data dictionary belonging to this file
        return my_data_dict

    def _load_attributes(self) -> None:
        # create the attribute lists stored in this class
        for _at in self._load_mappings_attrs:
            setattr(self, f'_{_at}', [])
        # load the data for the attributes lists and store it
        for _file_path_or_list in self._hdf5_path:
            self._load_specific_attributes(_file_path_or_list)

    @multimethod
    def _load_specific_attributes(self, _file_or_list: str) -> None:
        with h5.File(_file_or_list, 'r') as my_h5:
            for _d_i, (
                _d_i_path,
                _attr_pointer,
            ) in self._load_mappings_attrs.items():
                getattr(self, f'_{_d_i}').append(
                    my_h5[_d_i_path].attrs[_attr_pointer][()]
                )

    @multimethod
    def _load_specific_attributes(self, _file_or_list: list[str]) -> None:
        for _d_i, (
            _d_i_path,
            _attr_pointer,
        ) in self._load_mappings_attrs.items():
            _data_list = []
            for _fp in _file_or_list:
                with h5.File(_fp, 'r') as my_h5:
                    _data_list.append(my_h5[_d_i_path].attrs[_attr_pointer][()])
            getattr(self, f'_{_d_i}').append(_data_list)

    def _compute_frequency_data(
        self, og_shape: bool = True, additional_frequencies: bool = True
    ) -> None:
        # ASSUMES FREQUENCIES ARE STORED IN HZ --> TODO: generalize using frequency handler object!!!
        for _dd in self._list_data_dict:
            # corotating mode omegas
            _my_corot_omegas = _dd['corot_mode_omegas']
            # - parent modes
            _dd['corot_mode_omegas_p_muHz'] = hz_to_muhz(_my_corot_omegas[:, 0])
            _dd['corot_mode_omegas_p_cpd'] = hz_to_cpd(_my_corot_omegas[:, 0])
            # - daughter modes
            if og_shape:
                _dd['corot_mode_omegas_d_muHz'] = hz_to_muhz(
                    _my_corot_omegas[:, 1:]
                )
                _dd['corot_mode_omegas_d_cpd'] = hz_to_cpd(
                    _my_corot_omegas[:, 1:]
                )
            else:
                _dd['corot_mode_omegas_d_muHz'] = hz_to_muhz(
                    _my_corot_omegas[:, 1:].reshape(
                        -1,
                    )
                )
                _dd['corot_mode_omegas_d_cpd'] = hz_to_cpd(
                    _my_corot_omegas[:, 1:].reshape(
                        -1,
                    )
                )
            # compute the linear theoretical omega difference (corot omega)
            _dd['linear_omega_diff'] = (
                _my_corot_omegas[:, 0]
                - _my_corot_omegas[:, 1]
                - _my_corot_omegas[:, 2]
            )
            _dd['linear_omega_diff_muHz'] = hz_to_muhz(_dd['linear_omega_diff'])
            _dd['linear_omega_diff_cpd'] = hz_to_cpd(_dd['linear_omega_diff'])
            # compute the linear theoretical omega difference in terms of the minimum corot omega
            _min_corot_omega = np.abs(_my_corot_omegas).min(axis=1)
            _dd['linear_omega_diff_to_min_omega'] = (
                _dd['linear_omega_diff'] / _min_corot_omega
            )
            # compute and store the rotation frequency
            _my_inert_omegas = _dd['inert_mode_omegas']
            _my_rotation_freqs = (_my_inert_omegas - _my_corot_omegas)[
                :, 1
            ]  # m is 1 for this daughter mode!
            _dd['rotation_freqs'] = _my_rotation_freqs
            _dd['rotation_freqs_muHz'] = hz_to_muhz(_my_rotation_freqs)
            _dd['rotation_freqs_cpd'] = hz_to_cpd(_my_rotation_freqs)
            # spin parameter
            _dd['spin_parameters'] = (
                2.0 * _my_rotation_freqs[:, np.newaxis] / _my_corot_omegas
            )
            # dimensionless frequency ratios
            try:
                _dd['dimless_freq_ratio'] = (
                    _dd['inert_dimless_mode_omegas'] / _dd['inert_mode_omegas']
                )
            except KeyError:
                logger.exception(
                    'No dimensionless frequencies loaded, thus, I am not computing dimensionless frequency ratios.'
                )
            # compute additional frequency factors
            if additional_frequencies:
                # corotating mode frequencies
                _my_corot_freqs = _dd['corot_mode_freqs']
                # - parent modes
                _dd['corot_mode_freqs_p_muHz'] = hz_to_muhz(
                    _my_corot_freqs[:, 0]
                )
                _dd['corot_mode_freqs_p_cpd'] = hz_to_cpd(_my_corot_freqs[:, 0])
                # - daughter modes
                if og_shape:
                    _dd['corot_mode_freqs_d_muHz'] = hz_to_muhz(
                        _my_corot_freqs[:, 1:]
                    )
                    _dd['corot_mode_freqs_d_cpd'] = hz_to_cpd(
                        _my_corot_freqs[:, 1:]
                    )
                else:
                    _dd['corot_mode_freqs_d_muHz'] = hz_to_muhz(
                        _my_corot_freqs[:, 1:].reshape(
                            -1,
                        )
                    )
                    _dd['corot_mode_freqs_d_cpd'] = hz_to_cpd(
                        _my_corot_freqs[:, 1:].reshape(
                            -1,
                        )
                    )
                # inertial mode omegas
                _my_inert_omegas = _dd['inert_mode_omegas']
                # - parent modes
                _dd['inert_mode_omegas_p_muHz'] = hz_to_muhz(
                    _my_inert_omegas[:, 0]
                )
                _dd['inert_mode_omegas_p_cpd'] = hz_to_cpd(
                    _my_inert_omegas[:, 0]
                )
                # - daughter modes
                if og_shape:
                    _dd['inert_mode_omegas_d_muHz'] = hz_to_muhz(
                        _my_inert_omegas[:, 1:]
                    )
                    _dd['inert_mode_omegas_d_cpd'] = hz_to_cpd(
                        _my_inert_omegas[:, 1:]
                    )
                else:
                    _dd['inert_mode_omegas_d_muHz'] = hz_to_muhz(
                        _my_inert_omegas[:, 1:].reshape(
                            -1,
                        )
                    )
                    _dd['inert_mode_omegas_d_cpd'] = hz_to_cpd(
                        _my_inert_omegas[:, 1:].reshape(
                            -1,
                        )
                    )
                # inertial mode frequencies
                _my_inert_freqs = _dd['inert_mode_freqs']
                # - parent modes
                _dd['inert_mode_freqs_p_muHz'] = hz_to_muhz(
                    _my_inert_freqs[:, 0]
                )
                _dd['inert_mode_freqs_p_cpd'] = hz_to_cpd(_my_inert_freqs[:, 0])
                # - daughter modes
                if og_shape:
                    _dd['inert_mode_freqs_d_muHz'] = hz_to_muhz(
                        _my_inert_freqs[:, 1:]
                    )
                    _dd['inert_mode_freqs_d_cpd'] = hz_to_cpd(
                        _my_inert_freqs[:, 1:]
                    )
                else:
                    _dd['inert_mode_freqs_d_muHz'] = hz_to_muhz(
                        _my_inert_freqs[:, 1:].reshape(
                            -1,
                        )
                    )
                    _dd['inert_mode_freqs_d_cpd'] = hz_to_cpd(
                        _my_inert_freqs[:, 1:].reshape(
                            -1,
                        )
                    )

    def _compute_threshold_surface_luminosity_variations(self) -> None:
        for _dd in self._list_data_dict:
            # get a parametric resonance mask
            _parametric_mask = _dd['checks'][:, 0]
            # load individual luminosity amplitudes for the parent mode
            _surf_mode = _dd['surfs'][:, 0]
            # load theoretical amplitude for the parent mode
            _theory_1 = _dd['theoretical_amps'][:, 0]
            # get invalid amplitude mask
            _invalid_amplitude_mask = np.isnan(_theory_1) | (_theory_1 == 0.0)
            # compute luminosity conversion factor ONLY FOR PARAMETRIC RESONANCES (because they are only valid for such resonances; NaNs are used for non-parametric resonances)
            _lum_conv = np.divide(
                _surf_mode,
                _theory_1,
                where=_parametric_mask & ~_invalid_amplitude_mask,
                out=np.full_like(_surf_mode, np.nan),
            )
            # compute threshold amplitudes in ppm and store them
            _dd['thresh_surf_ppm'] = _dd['thresh_amp'] * _lum_conv * 1.0e6

    def _compute_driv_damp_info(self, og_shape: bool = True) -> None:
        for _dd in self._list_data_dict:
            # get driv_damp rates
            _driv_damp_rates = _dd['driv_damp_rates']
            # store parent linear driving rates separately
            _dd['p_driv_rates'] = _driv_damp_rates[:, 0]
            _dd['p_driv_rates_muHz'] = hz_to_muhz(_dd['p_driv_rates'])
            _dd['p_driv_rates_cpd'] = hz_to_cpd(_dd['p_driv_rates'])
            # store daughter linear damping rates separately
            if og_shape:
                _dd['d_damp_rates'] = _driv_damp_rates[:, 1:]
                _dd['d_damp_rates_muHz'] = hz_to_muhz(_dd['d_damp_rates'])
                _dd['d_damp_rates_cpd'] = hz_to_cpd(_dd['d_damp_rates'])
            else:
                _dd['d_damp_rates'] = _driv_damp_rates[:, 1:].reshape(
                    -1,
                )
                _dd['d_damp_rates_muHz'] = hz_to_muhz(_dd['d_damp_rates'])
                _dd['d_damp_rates_cpd'] = hz_to_cpd(_dd['d_damp_rates'])

    def _compute_amp_ratios(self) -> None:
        for _dd in self._list_data_dict:
            # get parametric mask
            _parametric_mask = _dd['checks'][:, 0]
            # get direct mask
            _direct_mask = _dd['checks'][:, 1]
            # load individual luminosity amplitudes
            _my_surfs = _dd['surfs']
            _surf_1 = _my_surfs[:, 0]
            _surf_2 = _my_surfs[:, 1]
            _surf_3 = _my_surfs[:, 2]
            # store luminosity amplitude ratios
            my_out_1 = np.full_like(_surf_2, np.nan)
            my_out_2 = np.full_like(_surf_3, np.nan)
            my_out_3 = np.full_like(_surf_3, np.nan)
            # get parametric daughter-parent ratios
            compute_single_amplitude_ratio(
                _surf_2,
                _surf_1,
                specified_out=my_out_1,
                additional_mask=_parametric_mask,
            )
            compute_single_amplitude_ratio(
                _surf_3,
                _surf_1,
                specified_out=my_out_2,
                additional_mask=_parametric_mask,
            )
            # get direct daughter-parent ratios (EVEN THOUGH THERE ARE NO THREE-MODE STATIONARY SOLUTIONS)
            compute_single_amplitude_ratio(
                _surf_1,
                _surf_2,
                specified_out=my_out_1,
                additional_mask=_direct_mask,
            )
            compute_single_amplitude_ratio(
                _surf_1,
                _surf_3,
                specified_out=my_out_2,
                additional_mask=_direct_mask,
            )
            # get biased ratios
            compute_single_amplitude_ratio(
                _surf_1,
                _surf_2 * _surf_3,
                specified_out=my_out_3,
                additional_mask=_parametric_mask,
            )
            compute_single_amplitude_ratio(
                _surf_1,
                _surf_2 * _surf_3,
                specified_out=my_out_3,
                additional_mask=_direct_mask,
            )
            # store both unbiased ratios
            _dd['surf_ratio_daughter_parent_OG_SHAPE'] = np.concatenate(
                (my_out_1.reshape(-1, 1), my_out_2.reshape(-1, 1)), axis=1
            )
            _dd['surf_ratio_daughter_parent'] = np.concatenate(
                [my_out_1, my_out_2]
            )
            # store ratio to compare with Van Beeck et al. (2021)
            _dd['surf_ratio_vb_21'] = my_out_3

    def _compute_cc_abs(self) -> None:
        for _dd in self._list_data_dict:
            # store the absolute value of the coupling coefficient and its energy-normalized form
            _dd['abs_kappa'] = np.abs(_dd['kappa'])
            _dd['abs_kappa_norm'] = np.abs(_dd['kappa_norm'])

    def _store_resonance_sharpness_factors(self) -> None:
        # ASSUMES FREQS STORED IN HZ
        for _dd in self._list_data_dict:
            # retrieve linear combination frequency
            _linear_combo = _dd['linear_omega_diff']
            # retrieve sum of linear daughter damping rates
            _my_damping = _dd['driv_damp_rates'][:, 1:].sum(axis=1)
            # compute and store resonance sharpness factor
            _dd['resonance_sharpness_factor'] = _linear_combo / _my_damping

    def _compute_freq_shifts(self, og_shape: bool = True) -> None:
        # ASSUMES FREQS STORED IN HZ
        for _dd in self._list_data_dict:
            # retrieve the linear combination frequency
            _my_linear_combo_freq = _dd['linear_omega_diff']
            if og_shape:
                _my_linear_combo_freq_double = np.tile(
                    _my_linear_combo_freq, (2, 1)
                ).T
            else:
                _my_linear_combo_freq_double = np.tile(_my_linear_combo_freq, 2)
            # retrieve the linear damping/driving rates
            _my_damping_driving = _dd['driv_damp_rates']
            _gamma_boxplus = _my_damping_driving.sum(axis=1)
            _gamma_p_ratio = -_my_damping_driving[:, 0] / _gamma_boxplus
            if og_shape:
                _gamma_d_ratio = (
                    _my_damping_driving[:, 1:] / _gamma_boxplus[:, np.newaxis]
                )
            else:
                _gamma_d_ratio = (_my_damping_driving[:, 1:]).reshape(
                    -1,
                ) / np.tile(_gamma_boxplus, 2)
            # compute and store the frequency shifts in different units
            _dd['freq_shift_p'] = _my_linear_combo_freq * _gamma_p_ratio
            _dd['freq_shift_p_muHz'] = hz_to_muhz(
                _my_linear_combo_freq * _gamma_p_ratio
            )
            _dd['freq_shift_p_cpd'] = hz_to_cpd(
                _my_linear_combo_freq * _gamma_p_ratio
            )
            _dd['freq_shift_d'] = _my_linear_combo_freq_double * _gamma_d_ratio
            _dd['freq_shift_d_muHz'] = hz_to_muhz(
                _my_linear_combo_freq_double * _gamma_d_ratio
            )
            _dd['freq_shift_d_cpd'] = hz_to_cpd(
                _my_linear_combo_freq_double * _gamma_d_ratio
            )
            # compute and store the frequency shift as a fraction of the mode frequency
            _dd['freq_shift_p_to_omega'] = (
                _dd['freq_shift_p_muHz'] / _dd['inert_mode_omegas_p_muHz']
            )
            _dd['freq_shift_d_to_omega'] = (
                _dd['freq_shift_d_muHz'] / _dd['inert_mode_omegas_d_muHz']
            )

    # unpack the plotting dictionary
    def _unpack_plotting_dict(self, plot_dict):
        """Internal initialization method that unpacks the plotting dictionary, and sets the corresponding attributes in the plotting object.

        Parameters
        ----------
        plot_dict : dict
            The dictionary that needs to be unpacked for plotting.
        """
        for _key, _val in plot_dict.items():
            setattr(self, f'_{_key}', _val)

    # generate the plotting masks based on loaded data
    def _generate_plotting_masks_sharpness(self) -> None:
        # initialize the lists containing number of isolated and non-isolated counts
        self._count_nr_dict = {'isolated': [], 'not isolated': []}
        # initialize the lists containing the masks
        self._mask_list_isolated = []
        self._mask_list_not_isolated_stable = []
        # generate the data for the plotting masks
        for _i, _dd in enumerate(self._list_data_dict):
            # create masks
            logger.debug(f'Performing masking operations for model {_i + 1}')
            (
                stringent_mask_isolated,
                stringent_mask_not_isolated,
            ) = isolated_non_isolated_stable_valid_masking(
                data_dict=_dd,
                minimum_threshold_lum=None,
                max_threshold_detuning=None,
            )
            self._mask_list_isolated.append(stringent_mask_isolated)
            self._mask_list_not_isolated_stable.append(
                stringent_mask_not_isolated
            )
            logger.debug(f'Masking operations done for model {_i + 1}')
            # store the number of counts
            logger.debug(f'Counting triads for model {_i + 1}')
            self._count_nr_dict['isolated'].append(
                stringent_mask_isolated.sum()
            )
            self._count_nr_dict['not isolated'].append(
                stringent_mask_not_isolated.sum()
            )
            logger.debug(f'Triad counting done for model {_i + 1}')

    # generate the resonance sharpness figure data
    def _generate_resonance_sharpness_figure_data(self) -> None:
        # create the zip object that will be used generate the figure data
        my_zipper = zip(
            self._mask_list_isolated,
            self._mask_list_not_isolated_stable,
            self._list_data_dict,
        )
        # initialize lists used to obtain masked data per dictionary
        isolated_sharpness_data_list = []
        not_isolated_sharpness_data_list = []
        # loop over data dictionaries to obtain the resonance sharpness figure data
        for _mi, _mni, _dd in my_zipper:
            # get the sharpness factor data
            _sharp = _dd['resonance_sharpness_factor']
            # append the isolated and non-isolated arrays
            isolated_sharpness_data_list.append(_sharp[_mi])
            not_isolated_sharpness_data_list.append(_sharp[_mni])
        # perform concatenation
        isolated_sharpness_data = np.concatenate(isolated_sharpness_data_list)
        not_isolated_sharpness_data = np.concatenate(
            not_isolated_sharpness_data_list
        )
        all_sharpness_data = np.concatenate(
            [isolated_sharpness_data, not_isolated_sharpness_data]
        )
        # store in the resonance sharpness figure data dict
        self._resonance_sharpness_figure_dict = {
            'all': all_sharpness_data,
            'isolated': isolated_sharpness_data,
            'not isolated': not_isolated_sharpness_data,
        }

    # compute the harmonic mask
    def _get_harmonic_mask(self) -> None:
        for _dd in self._list_data_dict:
            _ro = _dd['rad_ord']
            # compute the harmonic mask
            _dd['harmonic_mask'] = _ro[:, 1] == _ro[:, 2]

    # compute the minimal inertial frequency
    def _compute_minimal_inertial_frequencies(self):
        for _dd in self._list_data_dict:
            min_inertial_freq = _dd['inert_mode_freqs']
            min_inertial_freq = min_inertial_freq * 86400.0
            freq_shift_p = _dd['freq_shift_p_cpd'] / (2.0 * np.pi)
            freq_shifts_d = _dd['freq_shift_d_cpd'] / (2.0 * np.pi)
            min_inertial_freq[:, 0] = min_inertial_freq[:, 0] + freq_shift_p
            min_inertial_freq[:, 1:] = min_inertial_freq[:, 1:] + freq_shifts_d
            min_inertial_freq = min_inertial_freq.min(axis=1)
            # store the minimal inertial frequency
            _dd['min_inertial_freq_stack'] = np.column_stack(
                [min_inertial_freq, min_inertial_freq]
            )

    # perform the frequency mirroring
    def _perform_frequency_mirroring(self):
        # define Kepler Nyquist frequency  (Chaplin et al. 2014)
        kepler_notional_nyquist = (283.0 * 1e-6) * 86400.0
        for _dd in self._list_data_dict:
            # get link to the stack containing minimal frequencies
            my_minimal_frequency = _dd['min_inertial_freq_stack']
            # check for negatively shifted frequencies
            neg_mask = my_minimal_frequency < 0.0
            _my_abs = np.abs(my_minimal_frequency)
            # MIRROR THESE FREQUENCIES (i.e. show their frequency aliases instead; up to 4 times the notional Nyquist frequency of Kepler)
            # --- 1 mirroring operation ---
            one_mirror = (_my_abs < kepler_notional_nyquist) & neg_mask
            _dd['min_inertial_freq_stack'][one_mirror] = np.abs(
                my_minimal_frequency[one_mirror]
            )
            # --- 2 mirroring operations ---
            two_mirror = (
                (_my_abs < 2.0 * kepler_notional_nyquist)
                & (_my_abs >= kepler_notional_nyquist)
                & neg_mask
            )
            _dd['min_inertial_freq_stack'][two_mirror] = (
                2.0 * kepler_notional_nyquist
            ) + my_minimal_frequency[two_mirror]
            # --- 3 mirroring operations ---
            three_mirror = (
                (_my_abs < 3.0 * kepler_notional_nyquist)
                & (_my_abs >= 2.0 * kepler_notional_nyquist)
                & neg_mask
            )
            _dd['min_inertial_freq_stack'][three_mirror] = (
                -3.0 * kepler_notional_nyquist
            ) - my_minimal_frequency[three_mirror]
            # --- 4 mirroring operations ---
            four_mirror = (
                (_my_abs < 4.0 * kepler_notional_nyquist)
                & (_my_abs >= 3.0 * kepler_notional_nyquist)
                & neg_mask
            )
            _dd['min_inertial_freq_stack'][four_mirror] = (
                4.0 * kepler_notional_nyquist
            ) + my_minimal_frequency[four_mirror]
            # store the mask of negative values
            _dd['neg_mask'] = neg_mask

    # get the overview plot plotting masks
    def _get_overview_plotting_masks(self):
        # initialize the lists containing the masks
        self._mask_list_stable = []
        self._mask_list_ae_stable = []
        self._mask_list_stable_valid = []
        # fill the mask lists
        for _dd in self._list_data_dict:
            # compute the plotting masks in the specific delegated function
            (
                stable_mask,
                ae_stable_mask,
                stable_valid_mask,
            ) = theoretical_plotting_masks_overview(data_dict=_dd)
            # store these masks to generate the overview plot
            self._mask_list_stable.append(stable_mask)
            self._mask_list_ae_stable.append(ae_stable_mask)
            self._mask_list_stable_valid.append(stable_valid_mask)

    # calling method, which generates the plots
    def generate_plots(self):
        # create a resonance sharpness plot, if requested
        if self._plot_resonance_sharpness:
            logger.debug('Now plotting a resonance sharpness plot.')
            # initialize the sharpness figure object
            resonance_sharpness_object = ResonanceSharpnessFigure(
                sharpness_data_dict=self._resonance_sharpness_figure_dict,
                count_nr_dict=self._count_nr_dict,
                generic_save_dict=self._generic_save_data_dictionary,
                specific_save_dict=self._specific_save_data_dictionary,
            )
            # create the resonance sharpness plot using the gathered data
            resonance_sharpness_object.generate_plot(fig_size=self._fig_size)
            # save the figure(s)
            resonance_sharpness_object.save_plots(dpi=self._fig_dpi)
            logger.debug('Resonance sharpness plot generated.')
        # create an overview plot, if requested
        if self._plot_overview:
            logger.debug('Now plotting the overview plot.')
            # initialize the overview figure object, which gathers the necessary data for the plotting actions
            overview_figure_object = OverviewFigure(plotter=self)
            # generate the overview plot using the gathered data
            overview_figure_object.generate_plot(fig_size=self._fig_size)
            # save the figure(s)
            overview_figure_object.save_plots(dpi=self._fig_dpi)
            logger.debug('Overview plot generated.')
        # create a coupling coefficient profile plot, if necessary and possible
        if self._plot_coefficient_profile:
            # TODO: implement this, using saved profile data (these are not saved right now!)
            # log information if verbose
            logger.debug('Now plotting coupling coefficient profile.')
            # if self._solver_object._coupler is not None:
            #     self.coupling_coefficient_profile_plot()
            logger.debug('Coupling coefficient profile plot generated.')
