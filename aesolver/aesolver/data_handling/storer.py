"""Python module containing generic class used to store coupling data in efficient file formats.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import logging
from pathlib import Path
from multimethod import multimethod

# import from intra-package modules
from .unpacker import UnPacker
from .enumeration_files import SaveDirEnum, SaveSubDir
from .data_specific import HDF5Saver
from ..solver import QuadRotAESolver
from ..mode_input_generator import InputGen

# type checking imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


# set up logger
logger = logging.getLogger(__name__)


class NumericalSaver:
    """Class used to generate numerical output from data generated from nonlinear mode coupling models.

    Notes
    -----
    Can currently only save in HDF5 format.

    Parameters
    ----------
    base_dir: str, optional
        The path to the directory in which the output directory will be stored; by default '.' (or, make a save directory in the current directory).
    polytropic : bool, optional
        Denotes whether polytropic data is saved (True), or MESA+GYRE data (False); by default False.
    initial_save_format: str, optional
        Specifies which type of data is saved; by default 'hdf5', options: ['hdf5']
    """

    # attribute type declarations
    _store_poly: bool
    _base_directory: str
    _save_directory: str
    _save_format: str
    _solver: QuadRotAESolver
    _save_suffix: str
    _save_name: str
    _nr_modes: int
    _stat_sol_data: 'list[dict[str, Any] | None]'
    _couple_data: list[tuple]
    _coupling_stats: list[tuple]
    _nr_combos: int
    _f_hand: list[tuple]
    _mode_info: InputGen
    _num_int_method: str
    _use_cheby: bool
    _cheby_order_mul: int
    _cheby_blow_up: bool
    _cheby_blow_up_fac: float
    _unpacked_data: UnPacker

    def __init__(
        self,
        save_name: str,
        solving_object: QuadRotAESolver,
        base_dir: str = '.',
        save_dir: str = '',
        extra_specifier: str = '',
        polytropic: bool = False,
        initial_save_format: str = 'hdf5',
        nr_modes: int = 3,
    ) -> None:
        # store flag that is used to differentiate between
        # storage of polytropic data and MESA+GYRE data
        self._store_poly = polytropic
        # initialize the path to the base directory that
        # will hold the directory in which files will be saved
        self._base_directory = base_dir
        # initialize the path to the save directory as ''
        self._save_directory = ''
        # defines which type of data (initially) will be saved
        self._save_format = initial_save_format.lower()
        # store the solver object for now
        self._solver = solving_object
        # create the save directory and the save suffix
        self._create_save_directory_suffix()
        # generate the save file path
        self._generate_save_name(
            subdir=save_dir,
            save_name=save_name,
            extra_specifier=extra_specifier,
        )
        # store the number of modes considered in coupling
        self._nr_modes = nr_modes

    def __call__(self) -> None:
        # read the data to be stored from the solver object
        self._read_data_from_solver()
        # unpack data into object
        self._unpack_data_to_object()
        # initialize the data saver object
        _data_saver = HDF5Saver(
            self._save_name, self._unpacked_data, self._store_poly
        )
        # store/save these data
        self._save_data(_data_saver)

    @property
    def save_format(self) -> str:
        """Returns the format of data that will be saved.

        Returns
        -------
        str:
            Identifies the type of data to be saved.
        """
        return self._save_format

    @property
    def save_name(self) -> str:
        """Returns the path to the file that will be saved.

        Returns
        -------
        str:
            Path to the file that will be saved.
        """
        return self._save_name

    @property
    def unpacked_data(self) -> UnPacker:
        """Returns the unpacked data object used for saving to disk.

        Returns
        -------
        UnPacker:
            The unpacked data object.
        """
        return self._unpacked_data

    @save_format.setter
    def save_format(self, new_save_format: str) -> None:
        """Redefines the save format attribute, and (re)sets the save directory.

        Parameters
        ----------
        new_save_format: str
            Defines the new save format.
        """
        # change the save format
        self._save_format = new_save_format
        # create/(re)set the save directory
        self._create_save_directory_suffix()

    def _create_save_directory_suffix(self) -> None:
        """Internal utility method that generates the path to the save directory (based on `self._save_format') and creates that directory, if necessary.

        Notes
        -----
        Currently, the only supported save formats are: ['ascii', 'hdf5', 'csv'].
        """
        # generate the path to the save directory (using enumeration object) and the suffix of the save file
        (
            self._save_directory,
            self._save_suffix,
        ) = SaveDirEnum.obtain_save_directory(
            save_format=self._save_format, base_directory=self._base_directory
        )
        # create the directory, if necessary
        Path(f'{self._save_directory}').mkdir(parents=True, exist_ok=True)

    def _generate_save_name(
        self, subdir: str = '', save_name: str = '', extra_specifier: str = ''
    ) -> None:
        """Internal utility method that will generate the full path to the file used for saving.

        Parameters
        ----------
        subdir: str, optional
            Contains the name of the subdirectory, if necessary; by default empty string (no subdirectory!).
        save_name: str, optional
            Specifier for the name of the save file; by default empty string.
        extra_specifier: str, optional
            String that will be used to generate the information related to the frequency criterion used; by default empty string.

        Returns
        -------
        save_file_path: str
            The full path to the save file used for saving data.
        """
        # -- sub-directory
        # generate the sub-directory name based on a dataclass and its method
        _sub_directory_name = SaveSubDir(
            self._save_directory, subdir, extra_specifier=extra_specifier
        ).generate_subdirectory_path()
        # ensure the sub folder is generated
        Path(f'{_sub_directory_name}').mkdir(parents=True, exist_ok=True)
        # -- store full path/name of file
        self._save_name = (
            f'{_sub_directory_name}{save_name}.{self._save_suffix}'
        )

    def _read_data_from_solver(self) -> None:
        """Reads the data to be stored from the solver object."""
        # Read stationary solution data,
        # if data is not polytropic
        self._stat_sol_data = self._solver.stat_sol_info
        # Read coupling information data
        self._couple_data = self._solver.coupling_info
        # Read coupling statistics data
        self._coupling_stats = self._solver.coupling_statistics
        # Store the number of combinations and corresponding range
        self._nr_combos = len(self._stat_sol_data)
        # Store link to frequency data
        self._f_hand = self._solver.mode_freq_info
        # Store link to mode information data
        self._mode_info = self._solver._mode_info_object
        # store numerical integration method
        self._num_int_method = self._solver._numerical_integration_method
        # store angular integration kwargs
        self._use_cheby = self._solver._use_cheby
        self._cheby_order_mul = self._solver._cheby_order_mul
        self._cheby_blow_up = self._solver._cheby_blow_up
        self._cheby_blow_up_fac = self._solver._cheby_blow_up_fac

    def _unpack_data_to_object(self) -> None:
        """Unpack all necessary data for disk storage into an object/memory."""
        # initialize the unpacked data object
        self._unpacked_data = UnPacker(self)
        # unpack the data
        self._unpacked_data()
        # debug message
        logger.debug('Data unpacked into object!')

    @multimethod
    def _save_data(self, num_save_object: HDF5Saver) -> None:
        """Overloaded data saving method.

        Parameters
        ----------
        num_save_object : HDF5Saver
            The object containing the specific methods for saving HDF5 files.
        read_write_permissions : bool, optional
            If True, you gain full read-write permissions over the HDF5 file. If False, only append mode is used to add information to the HDF5 file!
        """
        # use the HDF5 saver object to store the data in a HDF5 file
        num_save_object.save_data()
        # log a message
        logger.info(f'HDF5 file saved at location {self.save_name}!')
