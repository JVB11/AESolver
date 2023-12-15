"""Python module containing class used to store numerical data related to nonlinear mode coupling in HDF5 format.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import logging
import re
import h5py as h5
import numpy as np

# import from intra-package modules
from .hdf5_specific import create_group_path_dataset, overwrite_dataset
from ..unpacker import UnPacker
from ..enumeration_files import Hdf5Open


# set up logger
logger = logging.getLogger(__name__)


# class that performs HDF5-specific save actions
class HDF5Saver:
    """Class performing HDF5-specific save actions for the nonlinear mode coupling data.

    Parameters
    ----------
    save_name : str
        The name/full path to the save file.
    unpacked_Data: UnPacker
        The unpacked data object.
    polytropic_save : bool
        If True, save data from polytrope computations. If False, save data from MESA+GYRE computations.
    """

    # attribute type declarations
    _store_poly: bool
    _file_path: str
    _unpacked_data: UnPacker
    _h5_mode: str
    _f_units: np.ndarray

    def __init__(
        self, save_name: str, unpacked_data: UnPacker, polytropic_save: bool
    ) -> None:
        # store whether polytropic or MESA+GYRE data are saved
        self._store_poly = polytropic_save
        # store the name/file path
        self._file_path = save_name
        # store the unpacked data
        self._unpacked_data = unpacked_data
        # perform operations to modify the save path to a name that reflects more which data are stored: add quantum number info
        self._adjust_file_path()
        # HDF5 file read/write/adapt mode, DEFAULT = write mode
        self._h5_mode = Hdf5Open.select_file_open_mode()
        # store the array of byte strings containing information on the units
        self._f_units = np.array(self._unpacked_data._f_units, dtype='S')

    @property
    def h5_mode(self) -> str:
        """Return the HDF5 read/write/append mode.

        Returns
        -------
        str
            Read/write/append mode for the HDF5 file.
        """
        return self._h5_mode

    @h5_mode.setter
    def h5_mode(self, bool_tup_mode: tuple[bool]) -> None:
        """Sets the HDF5 read/write/append mode.

        Parameters
        ----------
        bool_tup_mode : tuple[bool]
            Contains information to select the new file open mode.
        """
        self._h5_mode = Hdf5Open.select_file_open_mode(*bool_tup_mode)

    def _adjust_file_path(self) -> None:
        """Adjusts the file path to a name that incorporates quantum number information."""
        # construct the quantum number based string
        _qnr_string = f"_k_{'_'.join(self._unpacked_data._mode_k.astype(str))}_m_{'_'.join(self._unpacked_data._mode_m.astype(str))}"
        # substitute this string into the file path name
        self._file_path = re.sub(
            r'(.*\/\w+)(\.\w+)', rf'\1{_qnr_string}\2', self._file_path
        )

    def _create_base_attribute(
        self,
        h5file_stream: h5.File,
        attrname: str,
        value: np.ndarray | float | str | None = None,
    ) -> None:
        """Creates a file-level attribute.

        Parameters
        ----------
        h5file_stream : h5.File
            The stream to the HDF5 file.
        attrname : str
            The name of the attribute.
        value : np.ndarray | float | str, optional
            Value of the attribute, will store empty attribute if None; by default None.
        """
        if value is None:
            h5file_stream.attrs[attrname] = h5.Empty('f')
        else:
            h5file_stream.attrs[attrname] = value

    def _create_attribute_in_group(
        self,
        h5file_stream: h5.File,
        group_specifier: str,
        attrname: str,
        value: np.ndarray | float | str | None = None,
    ) -> None:
        """Creates the attribute in the group.

        Parameters
        ----------
        h5file_stream : h5.File
            The stream to the HDF5 file.
        group_specifier : str
            The specifier for the group path.
        attrname : str
            The name of the attribute.
        value : np.ndarray or float or str, optional
            Value of the attribute, will store empty attribute if None; by default None.
        """
        # get the group, if necessary
        h5group = create_group_path_dataset(h5file_stream, group_specifier)
        # check if an empty attribute needs to be made, or if a regular assignment is due
        if value is None:
            h5group.attrs[attrname] = h5.Empty('f')
        else:
            h5group.attrs[attrname] = value

    def _create_overwrite_dataset(
        self,
        h5file_stream: h5.File,
        group_specifier: str,
        dsetname: str,
        value: np.ndarray | None = None,
        dtype: type | None = None,
    ) -> None:
        """Creates/overwrites a new dataset in the HDF5-file.

        Parameters
        ----------
        h5file_stream : h5.File
            The stream to the HDF5 file.
        group_specifier : str
            The specifier for the group path.
        dsetname : str
            The name of the dataset.
        value : np.ndarray | float | str, optional
            Value of the dataset, will store empty dataset if None; by default None.
        """
        # create the dataset path
        _dataset_path = f'{group_specifier}/{dsetname}'
        # verify whether this dataset exists, and if so, overwrite it if possible.
        if _dataset_path in h5file_stream:
            # check if file was opened in mode that allows overwriting data
            if self._h5_mode == 'r+':
                # - dataset already exists, overwrite it
                overwrite_dataset(
                    h5file_stream=h5file_stream,
                    dataset_path=_dataset_path,
                    overwrite_value=h5.Empty('f') if (value is None) else value,
                )  # empty or value
        else:
            # dataset does not exist, store dataset
            if dtype is None:
                h5file_stream.create_dataset(
                    name=_dataset_path,
                    data=h5.Empty('f') if (value is None) else value,
                )  # empty or value
            else:
                h5file_stream.create_dataset(
                    name=_dataset_path,
                    dtype=dtype,
                    data=h5.Empty('f') if (value is None) else value,
                )  # empty or value

    def _create_grouped_attributes(
        self,
        h5file_stream: h5.File,
        group_specifier: str,
        attrname_list: list[str],
        value_list: list[np.ndarray | float | str],
    ) -> None:
        """Creates the attributes in the group.

        Parameters
        ----------
        h5file_stream : h5.File
            The stream to the HDF5 file.
        group_specifier : str
            The specifier for the group path.
        attrname_list : list[str]
            The names of the attribute.
        value_list : list[np.ndarray or float or str]
            Values of the attributes.
        """
        if len(group_specifier) == 0:
            # BASE/FILE level attributes
            for _aname, _aval in zip(attrname_list, value_list):
                self._create_base_attribute(h5file_stream, _aname, value=_aval)
        else:
            # GROUP level attributes
            for _aname, _aval in zip(attrname_list, value_list):
                self._create_attribute_in_group(
                    h5file_stream, group_specifier, _aname, value=_aval
                )

    def _create_grouped_datasets(
        self,
        h5file_stream: h5.File,
        group_specifier: str,
        dsetname_list: list[str],
        value_list: list[np.ndarray | None],
        dtypes: list[type],
    ) -> None:
        """Creates the datasets in the group.

        Parameters
        ----------
        h5file_stream : h5.File
            The stream to the HDF5 file.
        group_specifier : str
            The specifier for the group path.
        dsetname_list : list[str]
            The names of the dataset.
        value_list : list[np.ndarray or float or str]
            Values of the datasets.
        dtypes : list[np.float64 | np.bool_]
            Lists the datatypes for the different datasets.
        """
        for _dsname, _dsval, _dtyp in zip(dsetname_list, value_list, dtypes):
            self._create_overwrite_dataset(
                h5file_stream, group_specifier, _dsname, _dsval, dtype=_dtyp
            )

    def save_data(self) -> None:
        """Saves the numerical data in the HDF5 file."""
        # open a file stream so that you can write data to the HDF5 file
        with h5.File(self._file_path, self._h5_mode) as _h5file:
            # create numerical integration attributes (in base group)
            self._create_grouped_attributes(
                _h5file,
                '',
                [
                    'num_int_method',
                    'use_cheby',
                    'cheby_order_mul',
                    'cheby_blow_up',
                    'cheby_blow_up_fac',
                ],
                [
                    self._unpacked_data._num_int_method,
                    self._unpacked_data._use_cheby,
                    self._unpacked_data._cheby_order_mul,
                    self._unpacked_data._cheby_blow_up,
                    self._unpacked_data._cheby_blow_up_fac,
                ],
            )
            # create quantum number attributes
            self._create_grouped_attributes(
                _h5file,
                'quantum_numbers',
                ['l', 'm', 'k'],
                [
                    self._unpacked_data._mode_l,
                    self._unpacked_data._mode_m,
                    self._unpacked_data._mode_k,
                ],
            )
            # create radial orders dataset in the quantum number group
            self._create_overwrite_dataset(
                _h5file,
                'quantum_numbers',
                'n',
                value=self._unpacked_data._mode_rad_orders,
            )
            # create mode statistics dataset in the mode statistics group
            self._create_overwrite_dataset(
                _h5file,
                'mode_statistics',
                'wu',
                value=self._unpacked_data._wu_rule,
            )
            # create checks dataset
            # - ONLY for MESA+GYRE models
            if not self._store_poly:
                self._create_overwrite_dataset(
                    _h5file,
                    'checks',
                    'pre_checks',
                    value=self._unpacked_data._pre_checks,
                )
            # create ADIABATIC mode frequency datasets and attributes for the mode frequency information group
            self._create_grouped_datasets(
                _h5file,
                'mode_frequency_info',
                [
                    'inert_mode_freqs',
                    'corot_mode_freqs',
                    'inert_mode_omegas',
                    'corot_mode_omegas',
                    'inert_dimless_mode_freqs',
                    'corot_dimless_mode_freqs',
                    'inert_dimless_mode_omegas',
                    'corot_dimless_mode_omegas',
                ],
                [
                    self._unpacked_data._inert_mode_freqs,
                    self._unpacked_data._corot_mode_freqs,
                    self._unpacked_data._inert_mode_omegas,
                    self._unpacked_data._corot_mode_omegas,
                    self._unpacked_data._inert_dimless_mode_freqs,
                    self._unpacked_data._corot_dimless_mode_freqs,
                    self._unpacked_data._inert_dimless_mode_omegas,
                    self._unpacked_data._corot_dimless_mode_omegas,
                ],
                dtypes=[np.float64] * 8,
            )
            self._create_attribute_in_group(
                _h5file,
                'mode_frequency_info',
                'frequency_units',
                value=self._f_units[0],
            )
            # create NONADIABATIC mode frequency datasets and attributes for the mode frequency information group
            # - ONLY for MESA+GYRE models
            if not self._store_poly:
                self._create_grouped_datasets(
                    _h5file,
                    'nad_mode_frequency_info',
                    [
                        'inert_mode_freqs',
                        'corot_mode_freqs',
                        'inert_mode_omegas',
                        'corot_mode_omegas',
                        'inert_dimless_mode_freqs',
                        'corot_dimless_mode_freqs',
                        'inert_dimless_mode_omegas',
                        'corot_dimless_mode_omegas',
                    ],
                    [
                        self._unpacked_data._inert_mode_freqs_nad,
                        self._unpacked_data._corot_mode_freqs_nad,
                        self._unpacked_data._inert_mode_omegas_nad,
                        self._unpacked_data._corot_mode_omegas_nad,
                        self._unpacked_data._inert_dimless_mode_freqs_nad,
                        self._unpacked_data._corot_dimless_mode_freqs_nad,
                        self._unpacked_data._inert_dimless_mode_omegas_nad,
                        self._unpacked_data._corot_dimless_mode_omegas_nad,
                    ],
                    dtypes=[np.float64] * 8,
                )
                self._create_attribute_in_group(
                    _h5file,
                    'nad_mode_frequency_info',
                    'frequency_units',
                    value=self._f_units[0],
                )
            # create rotation datasets
            self._create_grouped_datasets(
                _h5file,
                'rotation',
                ['spin_factors', 'surface_rotation_frequency'],
                [
                    self._unpacked_data._spin_factors,
                    self._unpacked_data._surf_rot_f,
                ],
                dtypes=[np.float64, np.float64],
            )
            # create linear driving datasets
            # - ONLY FOR MESA+GYRE models
            if not self._store_poly:
                self._create_grouped_datasets(
                    _h5file,
                    'linear_driving',
                    ['driving_rates', 'quality_factors'],
                    [
                        self._unpacked_data._driving_rates,
                        self._unpacked_data._quality_factors,
                    ],
                    dtypes=[np.float64, np.float64],
                )
            # create mode coupling datasets
            self._create_grouped_datasets(
                _h5file,
                'coupling/generic',
                [
                    '|kappa|',
                    '|kappa_norm|',
                    '|eta|',
                    'adiabatic',
                    'quadratic_terms',
                ],
                [
                    self._unpacked_data._abs_ccs,
                    self._unpacked_data._abs_nccs,
                    self._unpacked_data._abs_etas,
                    self._unpacked_data._adiabatic,
                    self._unpacked_data._cc_terms,
                ],
                dtypes=[np.float64] * 3 + [np.bool_, np.float64],
            )
            # create mode normalization datasets
            self._create_overwrite_dataset(
                _h5file,
                'normalization',
                'radial',
                value=self._unpacked_data._norm_factors,
            )
            # create stationary solution datasets
            # - ONLY FOR MESA+GYRE models
            if not self._store_poly:
                self._create_grouped_datasets(
                    _h5file,
                    'coupling/stationary_three_mode',
                    [
                        'detunings',
                        'gammas',
                        'critical_parent_amplitudes',
                        'stationary_relative_phases',
                        'stationary_relative_luminosity_phases',
                        'q_factors',
                        'critical_q_factors',
                        'equilibrium_theoretical_amplitudes',
                        'surface_flux_variations',
                    ],
                    [
                        self._unpacked_data._detunings,
                        self._unpacked_data._gammas,
                        self._unpacked_data._crit_parent_amp,
                        self._unpacked_data._stat_sols_rel_phase,
                        self._unpacked_data._stat_sols_rel_lum_phase,
                        self._unpacked_data._q_factors,
                        self._unpacked_data._critical_q_factors,
                        self._unpacked_data._eq_amps,
                        self._unpacked_data._surf_flux_vars,
                    ],
                    dtypes=[np.float64] * 9,
                )
