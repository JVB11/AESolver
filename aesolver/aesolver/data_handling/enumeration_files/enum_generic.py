"""Enumeration module containing enumeration objects used to perform generic tasks.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import logging
import sys
from enum import Enum
from dataclasses import dataclass


# set up logger
logger = logging.getLogger(__name__)


# generic save directory enumeration
class SaveDirectoryEnum(Enum):
    """Python enumeration class containing different options for the creation of a save directory in the NumericalSaver object."""

    # define the attributes containing patterns used to obtain
    # the correct save directory path and the logger strings
    my_ascii = (
        '{base_directory}ascii_output/',
        '{base_directory}/ascii_output/',
        'Ascii',
        'asc',
    )
    my_hdf5 = (
        '{base_directory}hdf5_output/',
        '{base_directory}/hdf5_output/',
        'HDF5',
        'h5',
    )
    my_csv = (
        '{base_directory}csv_output/',
        '{base_directory}/csv_output/',
        'Csv',
        'csv',
    )

    # create the class method that will be used to generate the save directory OR raise an error!
    @classmethod
    def obtain_save_directory(
        cls, save_format: str, base_directory: str
    ) -> tuple[str, str]:
        """Method used to generate the path to the save directory, if possible. Will raise an error if not possible.

        Parameters
        ----------
        save_format: str
            Denotes the save format.
        base_directory: str
            Path to the base directory.

        Returns
        -------
        str
            Path to the save directory.
        str
            Suffix for the save file.
        """
        # check if the save format is in the list of allowed formats
        if (
            _attribute_string := f'my_{save_format.lower()}'
        ) in cls.__members__:
            # it is an allowed format: obtain the attribute
            (
                formatter_string,
                formatter_string_alt,
                _log_string,
                save_suffix,
            ) = cls.__getitem__(_attribute_string).value
            # generate the save directory path, return it and log this event
            logger.debug(f'{_log_string} save directory created, if necessary.')
            # guard against creation of hidden folders
            if _attribute_string[-1] == '/':
                return formatter_string.format(
                    base_directory=base_directory
                ), save_suffix
            else:
                return formatter_string_alt.format(
                    base_directory=base_directory
                ), save_suffix
        else:
            # generate an error message and exit
            logger.error(
                'Unknown save type. Cannot create save directory. Will now exit.'
            )
            sys.exit()


# generic save sub-directory dataclass -- comes after save dir enum, so all save types are allowed! (no need to check)
@dataclass
class SaveSubDirectory:
    """Python dataclass containing machinery for the creation of the subdirectory path in the NumericalSaver object.

    Parameters
    ----------
    save_directory: str
        Path to the save directory.
    sub_directory: str
        Name of the subdirectory in which you want to store your files (subdir of save directory).
    extra_specifier: str
        Additional specifier of use in the creation of sub-directory paths.
    """

    # define the slots
    __slots__ = ['save_directory', 'sub_directory', 'extra_specifier']
    # define the dataclass attributes
    save_directory: str
    sub_directory: str
    extra_specifier: str

    # define the method that takes care of the subdirectory slashes
    def _slash_subdir(self) -> str:
        """Internal method used to generate the correct subdirectory string based on the stored subdirectory.

        Returns
        -------
        str
            The correct sub-directory string used to generate
            the subdirectory path.
        """
        match self.sub_directory:
            case '' | None:
                return self.sub_directory
            case str() as my_sub if my_sub[-1] == '/':
                return f'{self.sub_directory[:-1]}'
            case str() as my_sub if my_sub[0] == '/':
                return self.sub_directory
            case _:
                return f'/{self.sub_directory}'

    # define the method used to generate the path to the subdirectory
    def generate_subdirectory_path(self) -> str:
        """Method used to generate the subdirectory path in the NumericalSaver object.

        Returns
        -------
        str
            The path to the subdirectory.
        """
        # retrieve the correct sub-directory string for subdirectory path generation
        _correct_subdir_path = self._slash_subdir()
        # check if an extra specifier needs to be added
        if len(self.extra_specifier) > 0:
            # add extra specifier
            return f'{self.save_directory}{_correct_subdir_path}{self.extra_specifier}'
        else:
            # no specifier needed
            return f'{self.save_directory}{_correct_subdir_path}'
