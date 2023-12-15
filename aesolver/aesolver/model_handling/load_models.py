"""Python file containing class that handles how to load model GYRE/MESA model data.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import logging
import os
import sys
from astropy.constants import G, M_sun, R_sun  # type: ignore
from tqdm import tqdm

# import intra-package modules
# - handle dimensionalization and loading of MESA/GYRE/polytrope files
from .enumeration_files import (
    DimGyre,
    GYRESummaryFiles,
    GYREDetailFiles,
    MESAProfileFiles,
    PolytropeModelFiles,
    PolytropeOscillationModelFiles,
    ReImPoly,
)

# - handle reading of model data
from .read_models import ModelReader

# - handle selection of MESA/GYRE/polytrope files
from .subselection_classes import (
    GYREDetailSubSelector,
    GYRESummarySubSelector,
    MESAProfileSubSelector,
    PolytropeModelSubSelector,
    PolytropeOscillationModelSubSelector,
)

# - handle polytropic computations
from ..polytropic_computations import PaSi

# typ checking imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


# initialize logger
logger = logging.getLogger(__name__)


# define class that is used to load GYRE/MESA model data
class ModelLoader:
    """Python class that implements methods used to load GYRE/MESA model data.

    Parameters
    ----------
    base_dir: str or None, optional
        The path to the base directory in which the directories containing the GYRE/MESA output files exist. If specified, the 'gyre_output_dir' and 'mesa_output_dir' paths will be considered relative to this base directory. If None, the 'gyre_output_dir' and 'mesa_output_dir' paths will be considered absolute; by default None.
    mesa_output_dir: str or None, optional
        If None, the loader will not search for MESA output files. If specified, the loader may search for MESA output files and read them, if directed. Additionally, if both 'base_dir' and this parameter is specified, this parameter is a relative path; by default None.
    gyre_output_dir: str or None, optional
        If None, the loader will not search for GYRE output files. If specified, the loader may search for GYRE output files and read them, if directed. Additionally, if both 'base_dir' and this parameter is specified, this parameter is a relative path; by default None.
    infer_dir: bool, optional
        If 'base_dir' is specified, and 'mesa_output_dir' and/or 'gyre_output_dir' are None, and additionally this parameter is set to True, we attempt to infer the paths to the MESA output and GYRE output directories, based on semantics of the subdirectories of the base directory. If False, no such inference is performed; by default True.
    search_l : list[int], optional
        The spherical degrees of the modes that need to be loaded; by default [2,2,2].
    search_m : list[int], optional
        The azimuthal orders of the modes that need to be loaded; by default [0,0,0].
    search_n : list[int], optional
        The radial orders of the modes that need to be loaded; by default [20,22,25].
    gyre_detail_substring : str, optional
        Selection (sub)string contained in the GYRE detail output file names; by default 'detail'.
    gyre_summary_substring : str, optional
        Selection (sub)string contained in the GYRE summary output file names; by default 'summary'.
    mesa_profile_substrings : tuple[str], optional
        Tuple of selection substrings contained in the MESA profile output file names; by default ('profile','KIC6352430').
    unit_system : str, optional
        The unit system for the computations; by default 'CGS'.
    mesa_suffix : str, optional
        The suffix for the MESA output files: by default 'dat'.
    use_polytrope_data : bool
        If True, load a polytropic model. If False, load a regular MESA model; by default False.
    polytrope_model_output_dir : str or None
        If None, the loader will not search for polytrope model (GYRE) output files. If specified, the loader may search for polytrope model (GYRE) output files and read them, if directed. Additionally, if both 'base_dir' and this parameter are specified, this parameter is a relative path; by default None.
    polytrope_oscillation_output_dir : str or None
        If None, the loader will not search for polytrope oscillation model (GYRE) output files. If specified, the loader may search for polytrope oscillation model (GYRE) output files and read them, if directed. Additionally, if both 'base_dir' and this parameter are specified, this parameter is a relative path; by default None.
    polytrope_model_suffix : str, optional
        The suffix for the polytrope model files; by default 'h5'.
    polytrope_oscillation_substrings : str, optional
        Selection (sub)string contained in the polytrope oscillation information file name; by default 'detail'.
    polytrope_model_substrings : tuple[str], optional
        Tuple of selection (sub)strings contained in the polytrope model file name; by default ('polytrope','n_3_0').
    """

    # attribute type declaration
    _base_dir: str | None
    _use_polytrope_data: bool
    _mesa_file_paths: list[str] | None
    _gyre_file_paths: list[str] | None
    _polytrope_model_file_paths: list[str] | None
    _polytrope_oscillation_file_paths: list[str] | None
    _gyre_detail_helper: GYREDetailSubSelector
    _gyre_summary_helper: GYRESummarySubSelector
    _mesa_profile_helper: MESAProfileSubSelector
    _polytrope_model_helper: PolytropeModelSubSelector
    _polytrope_oscillation_helper: PolytropeOscillationModelSubSelector
    _gyre_detail_reader: ModelReader
    _gyre_summary_reader: ModelReader
    _mesa_profile_reader: ModelReader
    _polytrope_model_reader: ModelReader
    _polytrope_oscillation_reader: ModelReader
    _gyre_detail_substring: list[str] | str | None
    _gyre_summary_substring: str | None
    _mesa_profile_substrings: list[str] | None | tuple[str, ...]
    _polytrope_model_substrings: list[str] | None | tuple[str, ...]
    _polytrope_oscillation_substrings: list[str] | None | str
    _gyre_detail_data: "list[Any]"
    _gyre_summary_data: list
    _mesa_profile_data: list
    _polytrope_model_data: list
    _polytrope_oscillation_data: list
    _unit_system: str
    _nr_modes: int
    _mesa_dir: str | None
    _gyre_dir: str | None
    _polytrope_model_dir: str | None
    _polytrope_oscillation_dir: str | None

    def __init__(
        self,
        base_dir: str | None = None,
        mesa_output_dir: str | None = None,
        gyre_output_dir: str | None = None,
        infer_dir: bool = True,
        search_l: list[int] = [2, 2, 2],
        search_m: list[int] = [0, 0, 0],
        search_n: list[int] = [20, 22, 25],
        gyre_detail_substring: str = 'detail',
        unit_system: str = 'CGS',
        gyre_summary_substring: str = 'summary',
        mesa_suffix: str = 'dat',
        mesa_profile_substrings: tuple[str, ...] | list[str] | None = (
            'profile',
            'KIC6352430',
        ),
        use_polytrope_data: bool = False,
        polytrope_model_output_dir: str | None = None,
        polytrope_oscillation_output_dir: str | None = None,
        polytrope_model_suffix: str = 'h5',
        polytrope_oscillation_substrings: str = 'detail',
        polytrope_model_substrings: tuple[str, ...] | list[str] | None = (
            'polytrope',
            'n_3_0',
        ),
    ) -> None:
        # store the base directory
        self._base_dir = base_dir
        # initialize the attribute that will determine which types of files shall be read
        self._use_polytrope_data = use_polytrope_data
        # store the MESA and GYRE subdirectory specifiers
        self._initialize_dirs(
            mesa=mesa_output_dir,
            gyre=gyre_output_dir,
            polytrope_model=polytrope_model_output_dir,
            polytrope_oscillation=polytrope_oscillation_output_dir,
        )
        # initialize the variables that will be used to store the paths
        # to the files, if necessary
        self._mesa_file_paths = None
        self._gyre_file_paths = None
        self._polytrope_model_file_paths = None
        self._polytrope_oscillation_file_paths = None
        # store whether directories shall be attempted to be inferred,
        # if no paths are specified
        self._infer_dirs(infer_dir=infer_dir)
        # store and initialize helper classes used to sub-select files
        self._gyre_detail_helper = GYREDetailSubSelector(
            search_l=search_l, search_m=search_m, search_n=search_n
        )
        self._gyre_summary_helper = (
            GYRESummarySubSelector()
        )  # TODO: maybe delete because it is not used?
        self._mesa_profile_helper = MESAProfileSubSelector(
            mesa_file_suffix=mesa_suffix
        )
        self._polytrope_model_helper = PolytropeModelSubSelector(
            polytrope_model_suffix=polytrope_model_suffix
        )
        self._polytrope_oscillation_helper = (
            PolytropeOscillationModelSubSelector(
                search_l=search_l, search_m=search_m, search_n=search_n
            )
        )
        # store and initialize the model reader classes
        self._gyre_detail_reader = ModelReader(
            enumeration_class=GYREDetailFiles  # type: ignore
        )
        self._gyre_summary_reader = ModelReader(
            enumeration_class=GYRESummaryFiles  # type: ignore
        )
        self._mesa_profile_reader = ModelReader(
            enumeration_class=MESAProfileFiles  # type: ignore
        )
        self._polytrope_model_reader = ModelReader(
            enumeration_class=PolytropeModelFiles  # type: ignore
        )
        self._polytrope_oscillation_reader = ModelReader(
            enumeration_class=PolytropeOscillationModelFiles  # type: ignore
        )
        # store the initial substrings to be checked
        self._gyre_detail_substring = gyre_detail_substring
        self._gyre_summary_substring = (
            gyre_summary_substring  # TODO: maybe delete because it is not used?
        )
        self._mesa_profile_substrings = mesa_profile_substrings
        self._polytrope_model_substrings = polytrope_model_substrings
        self._polytrope_oscillation_substrings = (
            polytrope_oscillation_substrings
        )
        # initialize the lists that will hold data
        self._gyre_detail_data = []
        self._gyre_summary_data = []
        self._mesa_profile_data = []
        self._polytrope_model_data = []
        self._polytrope_oscillation_data = []
        # store the unit system (to be used for possible unit conversion)
        self._unit_system = unit_system
        # store the number of modes for which data will be read
        self._nr_modes = len(search_l)

    # define the attribute getters/class properties
    @property
    def gyre_details(self) -> "list[Any]":
        """Property method for the gyre detail data, returning the data read from the GYRE detail files.

        Returns
        -------
        list[Any]
            Contains the data read from the GYRE detail files in dictionaries, for each of the selected modes.
        """
        return self._gyre_detail_data

    @property
    def gyre_summary(self) -> "list[Any]":
        """Property method for the gyre summary data, returning the data read from the GYRE summary file.

        Returns
        -------
        list[Any]
            Contains the data read from the GYRE summary file in dictionaries.
        """
        return self._gyre_summary_data

    @property
    def mesa_profile(self) -> "list[Any]":
        """Property method for the MESA profile data, returning the data read from the MESA profile files.

        Returns
        -------
        list[Any]
            Contains the data read from the MESA profile files in dictionaries.
        """
        return self._mesa_profile_data

    @property
    def polytrope_model(self) -> "list[Any]":
        """Property method for the polytrope model data, returning the data read from the polytrope model.

        Returns
        -------
        list[Any]
            Contains the data read from the polytrope model file in dictionaries.
        """
        return self._polytrope_model_data

    @property
    def polytrope_oscillation_models(self) -> "list[Any]":
        """Property method for the polytrope oscillation model data, returning the data read from the polytrope oscillation models.

        Returns
        -------
        list[Any]
            Contains the data read from the polytrope oscillation models in dictionaries.
        """
        return self._polytrope_oscillation_data

    # utility method that initializes directory paths
    def _init_dir_path(
        self, subdir: str | None, infer_subdir: bool
    ) -> str | None:
        """Utility internal method that generates a path based on input specifiers.

        Parameters
        ----------
        subdir : str | None
            Absolute or relative path to the directory containing the files of interest.
        infer_subdir : bool
            If True, override the default choosing options and always infer the path to the directory of interest based on the base directory.

        Returns
        -------
        str | None
            The path to the directory that is of interest. (If None, the path could not be resolved.)
        """
        # choose the correct return option
        if subdir is None:
            return subdir
        elif (self._base_dir is not None) and (not infer_subdir):
            return subdir
        else:
            return f'{self._base_dir}{subdir}'

    # method that initializes the MESA/GYRE directory paths, based on initialization values
    def _initialize_dirs(
        self,
        mesa: str | None,
        gyre: str | None,
        polytrope_model: str | None,
        polytrope_oscillation: str | None,
        infer_gyre: bool = False,
        infer_mesa: bool = False,
        infer_poly_struct: bool = False,
        infer_poly_modes: bool = False,
    ) -> None:
        """Internal method that Initializes the MESA and GYRE output directories, and the polytrope output directories, if possible.

        Parameters
        ----------
        mesa : str | None
            Absolute or relative path to the directory containing the MESA files.
        gyre : str | None
            Absolute or relative path to the directory containing the GYRE files.
        polytrope_model : str | None
            Absolute or relative path to the directory containing the polytrope model files.
        polytrope_oscillation : str | None
            Absolute or relative path to the directory containing the polytrope oscillation model files.
        infer_gyre : bool, optional
            If True, override the default choosing options and always infer the gyre directory from the base directory; by default False.
        infer_mesa : bool, optional
            If True, override the default choosing options and always infer the mesa directory from the base directory; by default False.
        infer_poly_struct : bool, optional
            If True, override the default choosing options and always infer the polytrope directories from the base directory; by default False.
        infer_poly_modes : bool, optional
            If True, override the default choosing options and always infer the polytrope directories from the base directory; by default False.
        """
        # verify if absolute or relative paths are specified at initialization, and initialize the attributes
        self._mesa_dir = self._init_dir_path(
            subdir=mesa, infer_subdir=infer_mesa
        )
        self._gyre_dir = self._init_dir_path(
            subdir=gyre, infer_subdir=infer_gyre
        )
        self._polytrope_model_dir = self._init_dir_path(
            subdir=polytrope_model, infer_subdir=infer_poly_struct
        )
        self._polytrope_oscillation_dir = self._init_dir_path(
            subdir=polytrope_oscillation, infer_subdir=infer_poly_modes
        )

    # internal method that logs and stores the results of the directory inference
    def _results_infer(self, list_dirs: list[str], type_specifier: str) -> None:
        """Internal workhorse method that updates the attributes based on the results of the directory inference.

        Parameters
        ----------
        list_dirs : list[str]
            Contains the possible inferred directory paths.
        type_specifier : str
            The type of directory whose path you are trying to infer.
        """
        # construct the attribute name
        _attr_name = f'_{type_specifier.lower()}_dir'
        # retrieve the attribute that needs to be updated
        if len(list_dirs) == 1:
            # update the directory attribute
            setattr(self, _attr_name, list_dirs[0])
            # log that you inferred the dir path
            logger.debug(
                f'Inferred the {type_specifier} directory path: {getattr(self, _attr_name)}'
            )
        elif len(list_dirs) == 0:
            # log that no directories were found
            logger.warning(
                f'No {type_specifier} directories could be inferred/found.'
            )
        else:
            # log that multiple directories were found
            logger.warning(
                f'Multiple possible {type_specifier} directories were found. ({list_dirs})\nAutomatic inference could not be done. Please restart and specify the exact directory, if you really need to read the {type_specifier} file(s).'
            )

    # method that infers directory paths if necessary
    def _infer_dirs(self, infer_dir: bool) -> None:
        """Internal method that infers directory paths

        Parameters
        ----------
        infer_dir : bool
            If True, try to infer the directory paths, if necessary. If False, no inference is performed.
        """
        if infer_dir and (
            (self._mesa_dir is None)
            or (self._gyre_dir is None)
            or (self._polytrope_model_dir is None)
            or (self._polytrope_oscillation_dir is None)
        ):
            logger.debug(
                'One of the MESA or GYRE or polytrope directory paths was uninitialized. Now attempting to infer that path based on semantics.'
            )
            # ensure that the base directory is specified
            if self._base_dir is None:
                logger.error(
                    'No path to the base directory is specified. Cannot infer any directories. Now exiting.'
                )
                sys.exit()
            # case: MESA directory unspecified
            if self._mesa_dir is None:
                # scan the directory for any subdirectories containing the name 'MESA'/'mesa'
                _possible_mesa_dirs = [
                    _d.path
                    for _d in os.scandir(self._base_dir)
                    if (_d.is_dir()) and ('mesa' in _d.name.lower())
                ]
                # ensure only one directory is found, and store the results
                self._results_infer(_possible_mesa_dirs, 'MESA')
            # case: GYRE directory unspecified
            if self._gyre_dir is None:
                # scan the directory for any subdirectories containing the name 'GYRE'/'gyre'
                _possible_gyre_dirs = [
                    _d.path
                    for _d in os.scandir(self._base_dir)
                    if (_d.is_dir()) and ('gyre' in _d.name.lower())
                ]
                # ensure only one directory is found, and store the results
                self._results_infer(_possible_gyre_dirs, 'GYRE')
            # case: polytrope model directory unspecified
            if self._polytrope_model_dir is None:
                # scan the directory for any subdirectories containing the name 'polytrope'
                _possible_polytrope_dirs = [
                    _d.path
                    for _d in os.scandir(self._base_dir)
                    if (_d.is_dir()) and ('polytrope_model' in _d.name.lower())
                ]
                # ensure only one directory is found, and store the results
                self._results_infer(_possible_polytrope_dirs, 'polytrope_model')
            # case: polytrope oscillation directory unspecified
            if self._polytrope_oscillation_dir is None:
                # scan the directory for any subdirectories containing the name 'polytrope'
                _possible_polytrope_dirs = [
                    _d.path
                    for _d in os.scandir(self._base_dir)
                    if (_d.is_dir())
                    and ('polytrope_oscillation' in _d.name.lower())
                ]
                # ensure only one directory is found, and store the results
                self._results_infer(
                    _possible_polytrope_dirs, 'polytrope_oscillation'
                )

    # method used to perform unit conversion
    @staticmethod
    def _unit_conversion(
        my_data: list[dict],
        my_unit_converter_class: type[GYRESummaryFiles]
        | type[GYREDetailFiles]
        | type[MESAProfileFiles],
        unit_system: str,
    ) -> list[dict]:
        """Internal method used to perform unit conversion.

        Parameters
        ----------
        my_data : list[dict]
            Contains the data that possibly need to be converted.
        my_unit_converter_class : GYRESummaryFiles | GYREDetailFiles |MESAProfileFiles
            Contains the unit conversion functions.
        unit_system : str
            The unit system to which units shall be converted, if necessary.

        Returns
        -------
        list[dict]
            Contains the unit-converted data.
        """
        # get the unit conversion function dictionary
        _conversion_dict = my_unit_converter_class.unit_conversions(
            unit_system=unit_system
        )
        # use the conversion dict to recreate the data dictionary, with updated values
        return [
            {_k: _conversion_dict[_k](_v) for _k, _v in _dat.items()}
            for _dat in my_data
        ]

    # callable method used to perform file sub-selection + reading data
    def read_data_files(
        self,
        gyre_detail: bool = False,
        gyre_summary: bool = False,
        mesa_profile: bool = False,
        polytrope_model: bool = False,
        polytrope_oscillation: bool = False,
        g_modes: list[bool] = [True, True, True],
        polytrope_mass: float = 3.0 * M_sun.cgs.value,
        polytrope_radius: float = 4.5 * R_sun.cgs.value,
        progress_bars: bool = False,
        progress_specifier: str = '',
    ) -> None:
        """Method used to read the data files, and load/store the data.

        Parameters
        ----------
        gyre_detail : bool, optional
            If True, sub-select GYRE detail files. Default: False.
        gyre_summary: bool, optional
            If True, sub-select GYRE summary files. Default: False.
        mesa_profile : bool, optional
            If True, sub-select MESA profiles. Default: False.
        polytrope_model : bool, optional
            If True, sub-select polytrope model files. Default: False.
        polytrope_oscillation : bool, optional
            If True, sub-select polytrope oscillation model files; by default False.
        g_modes : list[bool], optional
            Denotes which of the modes is a g mode. True if a g mode, False if not; by default [True,True,True].
        polytrope_mass : float, optional
            The mass of the polytrope of which you are loading data; by default 3.0 solar masses (in cgs units).
        polytrope_radius : float, optional
            The radius of the polytrope of which you are loading data; by default 4.5 solar radii (in cgs units).
        """
        # compute the value of the gravitational constant
        if self._unit_system.lower() == 'si':
            _g_val = G.si.value
        else:
            _g_val = G.cgs.value
        # read and subselect GYRE detail files, if necessary
        if gyre_detail:
            # log that you are sub-selecting based on gyre detail files
            logger.debug('Now sub-selecting GYRE detail files.')
            # run the data file reader method
            if progress_bars:
                self._gyre_detail_data = [
                    my_data_list[0]
                    for _i in tqdm(range(self._nr_modes), desc=f'Loading{f" {progress_specifier}" if len(progress_specifier) > 0 else ""} GYRE detail file data')
                    if (
                        len(
                            my_data_list
                            := self._gyre_detail_reader.data_file_reader(
                                my_dir=self._gyre_dir,
                                subselection_method=self._gyre_detail_helper.subselect,  # type: ignore (kwargs do no good for the typing!)
                                additional_substring=self._gyre_detail_substring,
                                mode_number=_i + 1,
                                g_mode=g_modes[_i],
                            )
                        )
                        == 1
                    )
                ]                
            else:
                self._gyre_detail_data = [
                    my_data_list[0]
                    for _i in range(self._nr_modes)
                    if (
                        len(
                            my_data_list
                            := self._gyre_detail_reader.data_file_reader(
                                my_dir=self._gyre_dir,
                                subselection_method=self._gyre_detail_helper.subselect,  # type: ignore (kwargs do no good for the typing!)
                                additional_substring=self._gyre_detail_substring,
                                mode_number=_i + 1,
                                g_mode=g_modes[_i],
                            )
                        )
                        == 1
                    )
                ]
            # verify the 'number of modes' files have been loaded
            if len(self._gyre_detail_data) != self._nr_modes:
                logger.error(
                    f'{self._nr_modes} mode files could not be loaded. Now exiting.'
                )
                sys.exit()
            # log information
            logger.debug('GYRE detail file sub-selection done.')
            logger.debug('Now performing unit conversion, if necessary.')
            # convert units of star-related quantities, if necessary
            # FREQUENCIES HANDLED SEPARATELY!!!
            self._gyre_detail_data = self._unit_conversion(
                my_data=self._gyre_detail_data,
                my_unit_converter_class=GYREDetailFiles,
                unit_system=self._unit_system,
            )
            logger.debug('Unit conversion done.')
            # add the value of the gravitational constant + add dimensionalized quantities to the output
            for _g_detail_dat in self._gyre_detail_data:
                _g_detail_dat['G'] = _g_val
                # add dimensionalized quantities to the output
                DimGyre.dimensionalize(_g_detail_dat)
            logger.debug('Dimensionalization done.')
        # read and subselect MESA profiles, if necessary
        if mesa_profile:
            # log that you are sub-selecting based on MESA profiles
            logger.debug('Now sub-selecting MESA profiles.')
            # run the data file reader method
            self._mesa_profile_data = (
                self._mesa_profile_reader.data_file_reader(
                    my_dir=self._mesa_dir,
                    subselection_method=self._mesa_profile_helper.subselect,  # type: ignore (kwargs do no good for the typing!)
                    structure_file=True,
                    additional_substring=self._mesa_profile_substrings,
                )
            )
            # verify that data from a single file has been loaded
            if (len(self._mesa_profile_data) != 1) or (
                len(self._mesa_profile_data[0]) == 0
            ):
                logger.error(
                    'The MESA profile could not be loaded. Now exiting.'
                )
                sys.exit()
            # convert units if necessary
            self._mesa_profile_data = self._unit_conversion(
                my_data=self._mesa_profile_data,
                my_unit_converter_class=MESAProfileFiles,
                unit_system=self._unit_system,
            )
            # log information
            logger.debug('MESA profile file sub-selection done.')
        # read and subselect GYRE summary files, if necessary
        if gyre_summary:
            logger.error('Not implemented.')
        # read and subselect polytrope model files, if necessary
        if polytrope_model:
            # log that you are now sub-selecting based on polytrope model files
            logger.debug('Now sub-selecting polytrope model files.')
            # run the data file reader method
            self._polytrope_model_data = (
                self._polytrope_model_reader.data_file_reader(
                    my_dir=self._polytrope_model_dir,
                    structure_file=True,
                    subselection_method=self._polytrope_model_helper.subselect,  # type: ignore (kwargs do no good for the typing!)
                    additional_substring=self._polytrope_model_substrings,
                )
            )
            # verify that data from a single file has been loaded
            if (len(self._polytrope_model_data) != 1) or (
                len(self._polytrope_model_data[0]) == 0
            ):
                logger.error(
                    'The polytrope model profile could not be loaded. Now exiting.'
                )
                sys.exit()
            # add the value of the gravitational constant
            for _poly_mod in self._polytrope_model_data:
                _poly_mod['G'] = _g_val
            # perform polytropic computations and add structure data
            _comp_object = PaSi(
                model_info_dictionary_list=self._polytrope_model_data,
                model_mass=polytrope_mass,
                model_radius=polytrope_radius,
            )
            # merge the dictionaries: add additional structure data
            for _poly_mod in self._polytrope_model_data:
                _poly_mod |= _comp_object.property_dict
            # log information
            logger.debug('Polytrope model sub-selection done.')
        # read and subselect polytrope oscillation model files,
        # if necessary
        if polytrope_oscillation:
            # log that you are now sub-selecting based on polytrope model files
            logger.debug('Now sub-selecting polytrope oscillation model files.')
            # run the data file reader method
            if progress_bars:
                self._polytrope_oscillation_data = [
                    my_data_list[0]
                    for _i in tqdm(range(self._nr_modes), desc='Loading polytrope oscillation data')
                    if (
                        len(
                            my_data_list
                            := self._polytrope_oscillation_reader.data_file_reader(
                                my_dir=self._polytrope_oscillation_dir,
                                subselection_method=self._polytrope_oscillation_helper.subselect,  # type: ignore (kwargs do no good for the typing!)
                                additional_substring=self._polytrope_oscillation_substrings,
                                mode_number=_i + 1,
                                g_mode=g_modes[_i],
                            )
                        )
                        == 1
                    )
                ]                
            else:
                self._polytrope_oscillation_data = [
                    my_data_list[0]
                    for _i in range(self._nr_modes)
                    if (
                        len(
                            my_data_list
                            := self._polytrope_oscillation_reader.data_file_reader(
                                my_dir=self._polytrope_oscillation_dir,
                                subselection_method=self._polytrope_oscillation_helper.subselect,  # type: ignore (kwargs do no good for the typing!)
                                additional_substring=self._polytrope_oscillation_substrings,
                                mode_number=_i + 1,
                                g_mode=g_modes[_i],
                            )
                        )
                        == 1
                    )
                ]
            # verify the 'number of modes' files have been loaded
            if len(self._polytrope_oscillation_data) != self._nr_modes:
                logger.error(
                    f'{self._nr_modes} mode files could not be loaded. Now exiting.'
                )
                sys.exit()
            # add G, mass and radius to the dictionaries
            for _poly_osc in self._polytrope_oscillation_data:
                _poly_osc['G'] = _g_val
                _poly_osc['M_star'] = polytrope_mass
                _poly_osc['R_star'] = polytrope_radius
                # convert re_im values to proper values
                ReImPoly.dimensionalize(_poly_osc)
            # verify the 'number of modes' files have been loaded
            if len(self._polytrope_oscillation_data) != self._nr_modes:
                logger.error(
                    f'{self._nr_modes} mode files could not be loaded. Now exiting.'
                )
                sys.exit()
            # log information
            logger.debug('Polytrope oscillation model file sub-selection done.')
