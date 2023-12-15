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
from .data_files import HDF5MappingsArrays, HDF5MappingsAttributes
from .library_functions import get_mask_information_triad_counts
from .utility_functions import hz_to_cpd, hz_to_muhz, compute_single_amplitude_ratio

# type checking imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # TODO: implement generic frequency conversions using this module!
    # EnumGYREFreqConv
    from typing import TypedDict, NotRequired
    from collections.abc import Sequence


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
class QuadraticAEStationaryAnalyzer:
    """Class that handles the analysis of the stationary solutions of the
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
        analysis_dict,
        save_information,
        requested_frequency_unit='cpd',
        model_names=['model1', 'model2', 'model3', 'model4', 'model5', 'model6', 'model7']
    ) -> None:
        # store the path to the stored HDF5 file
        self._hdf5_path = generate_load_file_path_list(
            load_dict=load_information
        )
        # unpack and store the plotting dictionary's values
        self._unpack_analysis_dict(analysis_dict=analysis_dict)
        # store the requested frequency unit
        self._freq_unit = requested_frequency_unit
        # load and compute the data for the analysis
        self._load_and_compute_data()
        # store the model names
        self._model_names = model_names

    # loads the plot data during initialization
    def _load_and_compute_data(self) -> None:
        # get the load mappings (for HDF5 output)
        self._load_mappings_arrays = HDF5MappingsArrays().get_dict()
        # get the attr mappings (for HDF5 output)
        self._load_mappings_attrs = HDF5MappingsAttributes().get_dict()
        # load data for the stationary solution analysis
        self._load_analysis_data(self._hdf5_path)
        self._load_attributes()
        self._compute_frequency_data(additional_frequencies=True, og_shape=True)
        self._compute_threshold_surface_luminosity_variations()
        self._compute_driv_damp_info()
        self._compute_amp_ratios()
        self._compute_cc_abs()
        # compute frequency shifts
        self._compute_freq_shifts()
        # compute non-linear frequencies
        self._store_nonlinear_frequency_observables()
        # compute dimensionless validity estimators
        self._compute_dimensionless_validity_estimators()
        # generate and store the masks
        self._store_masks_for_analysis()

    # load data for the resonance sharpness and overview plots - single string
    @multimethod
    def _load_analysis_data(self, my_file_name_or_names: str) -> None:
        # load the file data and store it in the data dictionary
        self._list_data_dict = [
            self._load_specific_file_data(file_path=my_file_name_or_names)
        ]

    @multimethod
    def _load_analysis_data(
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
    def _load_analysis_data(
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
            # compute the quality factor of the parent mode
            _dd['p_quality_factors'] = _dd['corot_mode_omegas'][:, 0] / _driv_damp_rates[:, 0]
            # store daughter linear damping rates separately
            if og_shape:
                _dd['d_damp_rates'] = _driv_damp_rates[:, 1:]
                _dd['d_damp_rates_muHz'] = hz_to_muhz(_dd['d_damp_rates'])
                _dd['d_damp_rates_cpd'] = hz_to_cpd(_dd['d_damp_rates'])
                # store the summed driving-damping rates in cpd
                _dd['gamma_sum_cpd'] = _dd['p_driv_rates_cpd'] + _dd['d_damp_rates_cpd'].sum(axis=1)
            else:
                og_damp_rates = _driv_damp_rates[:, 1:]
                _dd['d_damp_rates'] = og_damp_rates.reshape(
                    -1,
                )
                _dd['d_damp_rates_muHz'] = hz_to_muhz(_dd['d_damp_rates'])
                _dd['d_damp_rates_cpd'] = hz_to_cpd(_dd['d_damp_rates'])
                # store the summed driving-damping rates in cpd
                _dd['gamma_sum_cpd'] = _dd['p_driv_rates_cpd'] + og_damp_rates.sum(axis=1)

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
            # - freq shift as a whole
            _nonlinear_freq_shift_omega = np.zeros((_dd['freq_shift_p_to_omega'].shape[0], 3), dtype=_dd['freq_shift_p_to_omega'].dtype)
            _nonlinear_freq_shift_omega[:, 0] = _dd['freq_shift_p_to_omega']
            _nonlinear_freq_shift_omega[:, 1:] = _dd['freq_shift_d_to_omega']
            _dd['nonlinear frequency shift/omega inert'] = _nonlinear_freq_shift_omega
            
    def _store_nonlinear_frequency_observables(self) -> None:
        for _dd in self._list_data_dict:
            # retrieve the relevant nonlinear frequency shifts
            _freq_shift_cpd_p = _dd['freq_shift_p_cpd']
            _freq_shift_cpd_d = _dd['freq_shift_d_cpd']
            # store the nonlinear frequencies
            _dd['nonlinear parent omega (c/d)'] = _dd['inert_mode_omegas_p_cpd'] + _freq_shift_cpd_p
            _dd['nonlinear daughter omega (c/d)'] = _dd['inert_mode_omegas_d_cpd'] + _freq_shift_cpd_d
            # retrieve and store the relevant nonlinear relative phases
            _dd['nonlinear phase'] = _dd['stationary_relative_luminosity_phase']
            # store the nonlinear omegas as a whole
            _nonlinear_omegas_cpd = np.zeros((_dd['nonlinear parent omega (c/d)'].shape[0], 3), dtype=_dd['nonlinear parent omega (c/d)'].dtype)
            _nonlinear_omegas_cpd[:, 0] = _dd['nonlinear parent omega (c/d)']
            _nonlinear_omegas_cpd[:, 1:] = _dd['nonlinear daughter omega (c/d)']
            _dd['nonlinear omegas (c/d)'] = _nonlinear_omegas_cpd
            
    def _compute_dimensionless_validity_estimators(self) -> None:
        for _dd in self._list_data_dict:
            # get the info to compute the dimensionless parameters
            my_parent_drivers = _dd['p_driv_rates_cpd']
            my_daughter_dampers = _dd['d_damp_rates_cpd']
            # compute the summed gamma and store it
            summed_daughters = my_daughter_dampers.sum(axis=1)
            my_gamma_sums = my_parent_drivers + summed_daughters
            _dd['gamma_sum'] = my_gamma_sums
            # get the detunings in cpd
            my_lin_omega_diffs = _dd['linear_omega_diff_cpd']
            # compute q values and store them in the data dictionary
            _dd['q'] = my_lin_omega_diffs / my_gamma_sums
            _dd['q1'] = my_lin_omega_diffs / my_parent_drivers
            _dd['q3'] = my_lin_omega_diffs / my_daughter_dampers[:, 1]
            # compute theta values
            _dd['theta2'] = -my_daughter_dampers[:, 0] / my_parent_drivers
            _dd['theta3'] = -my_daughter_dampers[:, 1] / my_parent_drivers
            _dd['theta23'] = my_daughter_dampers[:, 0] / my_daughter_dampers[:, 1]
            
    def _store_masks_for_analysis(self) -> None:
        for _dd in self._list_data_dict:
            # retrieve the masking information
            stab_mask, ae_mask, isolation_mask, val_mask, stab_val_mask = get_mask_information_triad_counts(data_dict=_dd)
            # store mask information
            _dd['stability_mask'] = stab_mask
            _dd['ae_mask'] = ae_mask
            _dd['isolation_mask'] = isolation_mask
            _dd['validity_mask'] = val_mask
            _dd['stability_and_validity_mask'] = stab_val_mask

    # unpack the plotting dictionary
    def _unpack_analysis_dict(self, analysis_dict):
        """Internal initialization method that unpacks the plotting dictionary,
        and sets the corresponding attributes in the plotting object.

        Parameters
        ----------
        plot_dict : dict
            The dictionary that needs to be unpacked for plotting.
        """
        for _key, _val in analysis_dict.items():
            setattr(self, f'_{_key}', _val)
    
    def generate_mode_triad_numbers_per_model(self) -> None:
        # compute counts per model
        for _model, _dd in zip(self._model_names, self._list_data_dict):
            print('\n\n')
            # get the number of mode triads based on stored gamma_sum_info
            _gamma_sum = _dd['gamma_sum']
            print(f'Total nr. of Triads ({_model}): {_gamma_sum.shape[0]}')
            # get the number of mode triads that satisfy gamma_sum < 0
            print(f'Nr. of triads with _g_box_dot_ok < 0 ({_model}): {_dd["checks"][:, 3].sum()}')
            # get number of mode triads for which the Quartic (linear) stability condition is ok
            print(f'Nr. of triads for which Quartic condition is fulfilled ({_model}): {_dd["checks"][:, 4].sum()}')
            # get number of mode triads for which the hyperbolic condition is fulfilled
            print(f'Nr. of triads fulfilling the hyperbolic condition ({_model}): {_dd["checks"][:, 5].sum()}')
            # retrieve the masking information
            stab_mask, ae_mask, isolation_mask, val_mask, stab_val_mask = get_mask_information_triad_counts(data_dict=_dd)
            # print count information for the different masks
            print(f'Nr. of triads fulfilling the AE validity condition ({_model}): {_dd["ae_mask"].sum()}')
            print(f'Nr. of triads fulfilling the isolation validity condition ({_model}): {_dd["isolation_mask"].sum()}')
            print(f'Nr. of triads fulfilling the stability conditions ({_model}): {_dd["stability_mask"].sum()}')
            print(f'Nr. of triads fulfilling the validity conditions ({_model}): {_dd["validity_mask"].sum()}')
            print(f'Nr. of triads fulfilling the stability and validity conditions ({_model}): {_dd['stability_and_validity_mask'].sum()}')
            print('\n\n')
            
    def generate_counts_q_quality_factor_1_condition(self) -> None:
        # compute counts per model
        _list_both_ratios_greater_than_1 = []
        _list_q_greater_quality_factor_1 = []
        _nr_triads = []
        _list_both_ratios_and_q_quality_factor_1 = []
        _list_both_ratios_and_not_q_quality_factor_1 = []
        _list_stab = []
        _list_ae = []
        _list_isolation = []
        _list_val = []
        _list_stab_val = []
        _list_ae_only_ratios = []
        _list_information_both = []
        _list_information_not_quality = []
        for _model, _dd in zip(self._model_names, self._list_data_dict):
            # get the amplitude ratios
            _amp_ratios = _dd['surf_ratio_daughter_parent_OG_SHAPE']
            # get the mask for triads whose amplitude ratios are both larger than 1
            _both_larger_mask = (_amp_ratios > 1.0).prod(axis=1).astype(np.bool_)
            # append the number of triads whose ratios are both larger than unity
            _list_both_ratios_greater_than_1.append(_both_larger_mask.sum())
            # get the q factor
            _q = _dd['q']
            # get the Q1 factor (i.e. quality factor 1)
            _quality_factor_1 = _dd['p_quality_factors']
            # compute the mask for the necessary but not sufficient condition
            _m = _q > _quality_factor_1
            # store the counts
            _counts = _m.sum()
            # print the counts
            print(f'The number of triads for q > Q1 ({_model}) is {_counts}')
            # append the counts to the overall list
            _list_q_greater_quality_factor_1.append(_counts)
            # append the total number of triads for this model
            _nr_triads.append(_dd['q'].shape[0])
            # now check how many triads simultaneously check both conditions
            _both_conditions = _m & _both_larger_mask
            _ratio_not_quality = ~_m & _both_larger_mask
            # append results
            _list_both_ratios_and_q_quality_factor_1.append(_both_conditions.sum())
            _list_both_ratios_and_not_q_quality_factor_1.append(_ratio_not_quality.sum())
            # get the masks, to obtain mask-specific information
            stab_mask = _dd['stability_mask']
            ae_mask = _dd['ae_mask']
            isolation_mask = _dd['isolation_mask']
            val_mask = _dd['validity_mask']
            stab_val_mask = _dd['stability_and_validity_mask']
            # get mask specific information
            _no_stab = ~stab_mask & _both_conditions
            _stab = stab_mask & _both_conditions
            _ae = ae_mask & _both_conditions
            _no_ae = ~ae_mask & _both_conditions
            _isol = isolation_mask & _both_conditions
            _no_isol = ~isolation_mask & _both_conditions
            _val = val_mask & _both_conditions
            _no_val = ~val_mask & _both_conditions
            _stab_val = stab_val_mask & _both_conditions
            _no_stab_val = ~stab_val_mask & _both_conditions
            # append information
            _list_stab.append((_stab.sum(), _no_stab.sum()))
            _list_ae.append((_ae.sum(), _no_ae.sum()))
            _list_isolation.append((_isol.sum(), _no_isol.sum()))
            _list_val.append((_val.sum(), _no_val.sum()))
            _list_stab_val.append((_stab_val.sum(), _no_stab_val.sum()))
            # both ratios and AE
            _ae_both_ratios = ae_mask & _both_larger_mask
            _not_ae_both_ratios = ~ae_mask & _both_larger_mask
            # append information
            _list_ae_only_ratios.append((_ae_both_ratios.sum(), _not_ae_both_ratios.sum()))
            # use mask to get information on amplitude ratios that do not satisfy q > Q1
            _q_info = _q[_ratio_not_quality]
            _quality_info = _quality_factor_1[_ratio_not_quality]
            _amp_info = _amp_ratios[_ratio_not_quality, :]
            _rad_ord_info = _dd['rad_ord'][_ratio_not_quality]
            # append that information
            _list_information_not_quality.append((_q_info, _quality_info, _amp_info, _model, _rad_ord_info))
            # use mask to get information on amplitude ratios that satisfy q > Q1
            _q_info = _q[_both_conditions]
            _quality_info = _quality_factor_1[_both_conditions]
            _amp_info = _amp_ratios[_both_conditions, :]
            _rad_ord_info = _dd['rad_ord'][_both_conditions]
            # append that information
            _list_information_both.append((_q_info, _quality_info, _amp_info, _model, _rad_ord_info))
        # show end results for all models
        print(f'The total number of triads for which q > Q1 (all {sum(_nr_triads)} models): {sum(_list_q_greater_quality_factor_1)}')
        print(f'The total number of triads for which both predicted daughter-parent amplitude ratios are larger than unity (all {sum(_nr_triads)} models): {sum(_list_both_ratios_greater_than_1)}')
        print(f'The total number of triads for which both predicted daughter-parent amplitude ratios are larger than unity and q > Q1 (all {sum(_nr_triads)} models): {sum(_list_both_ratios_and_q_quality_factor_1)}')
        print(f'The total number of triads for which both predicted daughter-parent amplitude ratios are larger than unity and q <= Q1 (all {sum(_nr_triads)} models): {sum(_list_both_ratios_and_not_q_quality_factor_1)}')
        print(f'The total number of stable triads for which both predicted daughter-parent amplitude ratios are larger than unity and q > Q1 (all {sum(_nr_triads)} models): {sum([x[0] for x in _list_stab])}')
        print(f'The total number of unstable triads for which both predicted daughter-parent amplitude ratios are larger than unity and q > Q1 (all {sum(_nr_triads)} models): {sum([x[1] for x in _list_stab])}')
        print(f'The total number of AE-valid triads for which both predicted daughter-parent amplitude ratios are larger than unity and q > Q1 (all {sum(_nr_triads)} models): {sum([x[0] for x in _list_ae])}')
        print(f'The total number of AE-invalid triads for which both predicted daughter-parent amplitude ratios are larger than unity and q > Q1 (all {sum(_nr_triads)} models): {sum([x[1] for x in _list_ae])}')
        print(f'The total number of isolated triads for which both predicted daughter-parent amplitude ratios are larger than unity and q > Q1 (all {sum(_nr_triads)} models): {sum([x[0] for x in _list_isolation])}')
        print(f'The total number of non-isolated triads for which both predicted daughter-parent amplitude ratios are larger than unity and q > Q1 (all {sum(_nr_triads)} models): {sum([x[1] for x in _list_isolation])}')
        print(f'The total number of valid triads for which both predicted daughter-parent amplitude ratios are larger than unity and q > Q1 (all {sum(_nr_triads)} models): {sum([x[0] for x in _list_val])}')
        print(f'The total number of invalid triads for which both predicted daughter-parent amplitude ratios are larger than unity and q > Q1 (all {sum(_nr_triads)} models): {sum([x[1] for x in _list_val])}')
        print(f'The total number of stable and valid triads for which both predicted daughter-parent amplitude ratios are larger than unity and q > Q1 (all {sum(_nr_triads)} models): {sum([x[0] for x in _list_stab_val])}')
        print(f'The total number of instable and/or invalid triads for which both predicted daughter-parent amplitude ratios are larger than unity and q > Q1 (all {sum(_nr_triads)} models): {sum([x[1] for x in _list_stab_val])}')
        print(f'The total number of AE-valid triads for which both predicted daughter-parent amplitude ratios are larger than unity (all {sum(_nr_triads)} models): {sum([x[0] for x in _list_ae_only_ratios])}')
        print(f'The total number of AE-invalid triads for which both predicted daughter-parent amplitude ratios are larger than unity (all {sum(_nr_triads)} models): {sum([x[1] for x in _list_ae_only_ratios])}')
        print('\n\n')
        print(f'Information on the triads that satisfy q > Q1 and have two predicted daughter-parent amplitude ratios larger than unity:')
        for _q, _qual, _amp, _mod, _rad in _list_information_both:
            if (q_len := len(_q)) > 0:
                print(f'{_mod.capitalize()} has {q_len} triads of this type.')
                print('Their properties are:')
                print(f'q: {_q}')
                print(f'Q1: {_qual}')
                print(f'Daughter-parent amplitude ratio: {_amp}')
                print(f'Radial orders: {_rad}')
            else:
                print(f'{_mod.capitalize()} has no triads of this type.')
        print('\n\n')
        print(f'Information on the triads that do not satisfy q > Q1 yet have two predicted daughter-parent amplitude ratios larger than unity:')
        for _q, _qual, _amp, _mod, _rad in _list_information_not_quality:
            if (q_len := len(_q)) > 0:
                print(f'{_mod.capitalize()} has {q_len} triads of this type.')
                print('Their properties are:')
                print(f'q: {_q}')
                print(f'Q1: {_qual}')
                print(f'Daughter-parent amplitude ratio: {_amp}')
                print(f'Radial orders: {_rad}')
            else:
                print(f'{_mod.capitalize()} has no triads of this type.')
        print('\n\n')

    def generate_isolated_triad_mode_properties(self) -> None:
        # get information per model
        for _model, _dd in zip(self._model_names, self._list_data_dict):
            print('\n\n')
            # get stability validity mask
            _stab_val = _dd['stability_and_validity_mask']
            # get necessary information for isolated triads
            print(f'Radial orders of isolated triads ({_model}): {_dd["rad_ord"][_stab_val]}')
            print(f'Delta Omega of isolated triads (cpd; {_model}): {_dd["linear_omega_diff_cpd"][_stab_val]}')
            print(f'Gamma_sum of isolated triads (10-3 cpd; {_model}): {_dd["gamma_sum_cpd"][_stab_val]*1000.0}')
            _p_om = _dd['corot_mode_omegas_p_cpd'][_stab_val]
            _d_om = _dd['corot_mode_omegas_d_cpd'][_stab_val]  # assumes OG shape = True
            print(f'Corot omegas of isolated triads (cpd; {_model}): ({_p_om}, {_d_om[:, 0]}, {_d_om[:, 1]})')
            _p_g = _dd['p_driv_rates_cpd'][_stab_val]
            _d_g = _dd['d_damp_rates_cpd'][_stab_val]  # assumes OG shape = True
            print(f'Gammas of isolated triads (10-5 cpd; {_model}): ({_p_g*100000.0}, {_d_g[:, 0]*100000.0}, {_d_g[:, 1]*100000.0})')
            _spins = _dd['spin_parameters'][_stab_val]
            _nan_max_spins = np.nanmax(_spins, axis=1)
            print(f'Maximal spin parameter of isolated triads ({_model}): {_nan_max_spins}')
            print('\n\n')
    
    def generate_isolated_triad_stab_val_estimators(self) -> None:
        # get information per model
        for _model, _dd in zip(self._model_names, self._list_data_dict):
            print('\n\n')
            # get stability validity mask
            _stab_val = _dd['stability_and_validity_mask']
            # get necessary information for isolated triads
            print(f'Radial orders of isolated triads ({_model}): {_dd["rad_ord"][_stab_val]}')
            print(f'AE parameters for isolated triads ({_model}): {np.abs(_dd["linear_omega_diff_to_min_omega"][_stab_val])}')
            print(f'|q| values for isolated triads ({_model}): {np.abs(_dd["q"][_stab_val])}')
            print(f'|q1| values for isolated triads ({_model}): {np.abs(_dd["q1"][_stab_val])}')
            print(f'Theta_2 values for isolated triads ({_model}): {_dd["theta2"][_stab_val]}')
            print(f'Theta_3 values for isolated triads ({_model}): {_dd["theta3"][_stab_val]}')
            print(f'O(|eta1|) values for isolated triads ({_model}): {_dd["abs_kappa_norm"][_stab_val]}')
            print('\n\n')
            
    def generate_isolated_triad_observables(self) -> None:
                # get information per model
        for _model, _dd in zip(self._model_names, self._list_data_dict):
            print('\n\n')
            # get stability validity mask
            _stab_val = _dd['stability_and_validity_mask']
            # get necessary information for isolated triads
            print(f'Radial orders of isolated triads ({_model}): {_dd["rad_ord"][_stab_val]}')
            # get the frequency shift to frequency ratio
            _rat = _dd['nonlinear frequency shift/omega inert'][_stab_val]
            _rat_max = np.nanmax(np.abs(_rat), axis=1)
            print(f'Maximal absolute non-linear frequency shifts as a fraction of the linear inertial-frame angular frequency (ppt; {_model}): {_rat_max*1000.0}')
            # get the relative phase
            print(f'Nonlinear combination phases ({_model}): {_dd["nonlinear phase"][_stab_val]}')
            # get surface fluctuation ratios
            _surf_rats = _dd['surf_ratio_daughter_parent_OG_SHAPE']
            print(f'Surface luminosity fluctuation ratios ({_model}): {_surf_rats[_stab_val, 0]} {_surf_rats[_stab_val, 1]}')
            print('\n\n')
