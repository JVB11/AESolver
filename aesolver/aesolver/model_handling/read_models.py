"""Python file containing class that handles how to load and read GYRE/MESA model data.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import logging
import os
import h5py as h5
import pandas as pd
from functools import reduce, partial
from operator import itemgetter

# import from intra-package modules
from .enumeration_files import EnumReadInfo

# type checking imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any, TypedDict, NotRequired
    from typing_extensions import Protocol

    # get enumeration classes for typing
    from .enumeration_files import (
        GYRESummaryFiles,
        GYREDetailFiles,
        MESAProfileFiles,
        PolytropeModelFiles,
        PolytropeOscillationModelFiles,
    )

    # define protocols for the sub-selection methods:
    # had to leave out the not-required mode_number and g_mode!
    class SubSelectProtocol(Protocol):
        def __call__(
            self,
            name: str,
            additional_substring: list[str] | str | None | tuple[str, ...],
            **kwargs: bool | int,
        ) -> bool:
            ...

    # define typed dictionaries for the dictionary used during sub-selection
    class SubSelectDict(TypedDict):
        additional_substring: None | str | list[str] | tuple[str, ...]
        mode_number: NotRequired[int]
        g_mode: NotRequired[bool]


# initialize logger
logger = logging.getLogger(__name__)


# function that composes several functions to obtain the suffix
def _compose_func_for_suffix(*fs: 'Callable') -> 'Callable[[str], str]':
    """Function composition function used in this module to find the suffix of a file.

    Parameters
    ----------
    fs : tuple[Callable, ...]
        The list of functions that need to be composed.

    Returns
    -------
    Callable[[str], str]
        The composed function used to find the file suffix.
    """
    return reduce(lambda f, g: lambda x: f(g(x)), fs)


class ModelReader:
    """Python class containing methods that are used to define how to
    read specific models.

    Parameters
    ----------
    enumeration_class: GYRESummaryFiles | GYREDetailFiles | MESAProfileFiles | PolytropeModelFiles | PolytropeOscillationModelFiles
        Enumeration class used to load the different attributes possibly stored within these files.
    """

    # attribute type declarations
    _suffix_finder: 'Callable'
    _enumeration_class: 'GYRESummaryFiles | GYREDetailFiles | MESAProfileFiles | PolytropeModelFiles | PolytropeOscillationModelFiles'

    def __init__(
        self,
        enumeration_class: 'GYRESummaryFiles | GYREDetailFiles | MESAProfileFiles | PolytropeModelFiles | PolytropeOscillationModelFiles',
    ) -> None:
        # store function used to infer the suffix of files
        self._suffix_finder = _compose_func_for_suffix(
            itemgetter(1), partial(str.rsplit, sep='.', maxsplit=1)
        )
        # store the enumeration class
        self._enumeration_class = enumeration_class

    # method used to read files, with the specifics depending on the file suffix
    def _read_file(self, file_path: str, file_suffix: str) -> 'dict[str, Any]':
        """Internal method used to read/extract information from GYRE/MESA files, with the implementation differing based on the file suffix.

        Parameters
        ----------
        file_path: str
            The path to the file to be read.
        file_suffix: str
            The suffix of the file.

        Returns
        -------
        dict[str, Any]
            Contains the information read from a specific file.
        """
        # obtain the list of attributes that may be read
        _possible_attributes = self._enumeration_class.attrs()
        # determine the method of action based on the file suffix
        reader = EnumReadInfo.reader(
            file_suffix=file_suffix, file_path=file_path
        )(self)
        # obtain and return the data that were read into a dictionary
        return reader(file_path=file_path, attributes=_possible_attributes)

    # method that handles how data from the dat files are read
    def _dat_reader(
        self, file_path: str, attributes: list[str]
    ) -> 'dict[str, Any]':
        """Internal method that handles how data from .txt files are read.

        Parameters
        ----------
        file_path: str
            The path to the file to be read.
        attributes: list[str]
            Contains the attributes that potentially are present in the HDF5 file.

        Returns
        -------
        my_data : dict[str, Any]
            The dictionary containing the data from the read.
        """
        # initialize the data dictionary
        my_data = {}
        # open the MESA profile.dat file and read into a dataframe
        _my_df = pd.read_fwf(file_path, skiprows=5)
        if TYPE_CHECKING:
            assert isinstance(_my_df, pd.DataFrame)
        # read the header information in a pandas series
        my_read = pd.read_fwf(file_path, skiprows=1, infer_nrows=2)
        if TYPE_CHECKING:
            assert isinstance(my_read, pd.DataFrame | pd.Series)
        _my_header_info = my_read.iloc[0]
        # loop over the attributes, and add them to the dictionary, if present
        for _possible_attr in attributes:
            # add to the dictionary
            try:
                # try reading regular dataframe info
                my_data[_possible_attr] = _my_df.loc[
                    :, _possible_attr
                ].to_numpy()
            except KeyError:
                # try reading header info
                try:
                    my_data[_possible_attr] = _my_header_info.at[_possible_attr]
                except KeyError:
                    logger.debug(
                        f'No valid key found, discarding the "{_possible_attr}" attribute field.'
                    )
        # return the data dictionary
        return my_data

    # method that handles how data from the text files are read
    def _text_reader(
        self, file_path: str, attributes: list[str]
    ) -> 'dict[str, Any]':
        """Internal method that handles how data from .txt files are read.

        Parameters
        ----------
        file_path: str
            The path to the file to be read.
        attributes: list
            Contains the attributes that potentially are present in the HDF5 file.

        Returns
        -------
        my_data : dict[str, Any]
            The dictionary containing the data from the read.
        """
        # initialize the data dictionary
        my_data = {}
        # TODO: implement this method
        logger.error('TO BE IMPLEMENTED')
        # return the data dictionary
        return my_data

    # method that handles how data from HDF5 files are read
    def _hdf5_reader(
        self, file_path: str, attributes: list[str]
    ) -> 'dict[str, Any]':
        """Internal method that handles how data from HDF5 files are read.

        Parameters
        ----------
        file_path: str
            The path to the file to be read.
        attributes: list
            Contains the attributes that potentially are present in the HDF5 file.

        Returns
        -------
        my_data : dict[str, Any]
            The dictionary containing the data from the read.
        """
        # initialize the data dictionary
        my_data = {}
        # open the HDF5 file, loop through the possible attributes and store them in the dictionary if found.
        with h5.File(file_path, 'r') as my_hdf5:
            # loop over the attributes, and add them to the dictionary, if present
            for _possible_attr in attributes:
                # add to the dictionary
                try:
                    hdf5_index = my_hdf5[_possible_attr]
                    if TYPE_CHECKING:
                        assert isinstance(hdf5_index, h5.Group | h5.Dataset)
                    my_data[_possible_attr] = hdf5_index[()]
                except KeyError:
                    try:
                        my_data[_possible_attr] = my_hdf5.attrs[_possible_attr]
                    except KeyError:
                        logger.debug(
                            f'No valid key found, discarding the "{_possible_attr}" attribute field.'
                        )
        # return the data dictionary
        return my_data

    # method that scans files in directory + selects and stores the requested files
    def data_file_reader(
        self,
        my_dir: str | None,
        subselection_method: 'SubSelectProtocol',
        additional_substring: str | list[str] | tuple[str, ...] | None = None,
        mode_number: int = 1,
        structure_file: bool = False,
        g_mode: bool | None = None,
    ) -> 'list[dict[str, Any]]':
        """Method used to construct the list of GYRE/MESA files requested for the coupling computations.

        Parameters
        ----------
        my_dir: str | None
            The directory in which files shall be sub-selected.
        subselection_method : SubSelectProtocol
            The method to be used for file subselection.
        additional_substring : str | list[str] | tuple[str, ...] | None, optional
            An additional substring used for additional sub-selection. (Needs to be present in the file name.) No sub-selection if None; by default None.
        mode_number : int, optional
            The number of the mode for which you are trying to read data files; by default 1.
        structure_file : bool, optional
            True when reading MESA profiles or polytrope model files (pure stellar structure files), False otherwise; by default False.
        g_mode : bool | None, optional
            If True, search for files containing g modes. If False, search for files containing other modes. If None, ignore this parameter; by default None.

        Returns
        -------
        data_list : list[dict[str, Any]]
            The list containing the data dictionaries.
        """
        # initialize list that will store the GYRE/MESA data
        data_list = []
        # determine the kwargs for the sub-selection
        _kwargs_sub: 'SubSelectDict'
        _kwargs_sub = {'additional_substring': additional_substring}
        if not structure_file:
            _kwargs_sub['mode_number'] = mode_number
        if g_mode is not None:
            _kwargs_sub['g_mode'] = g_mode
        # loop through the list of GYRE files and select the necessary files
        for _f in os.scandir(my_dir):
            # determine whether the file adheres to the preselected conditions
            if _f.is_file() and subselection_method(
                name=_f.name, **_kwargs_sub
            ):
                # obtain the suffix
                _suff = self._suffix_finder(_f.name)
                # read the file + append the file data to the list
                data_list.append(self._read_file(_f.path, _suff))
        # return the list containing the read data
        return data_list
