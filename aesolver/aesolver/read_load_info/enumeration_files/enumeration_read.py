"""Python module containing enumeration classes containing conversions of read-file comments to usable dictionary keys.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import logging
import sys
from enum import Enum

# relative import
from .super_enumeration_read import EnumerationDefaultSuperRead as EdsR


# set up logger
logger = logging.getLogger(__name__)


# enumeration class for conversion of information dictionary keys
class InfoDictConverter(Enum):
    """Enumeration dictionary containing information to convert comments read from file to dictionary keys."""

    # conversion dict
    CONVERSION_DICT = {
        'MESA base folder/directory': 'base_dir',
        'MESA default output folder/directory': 'mesa_output_dir',
        'MESA default profile substrings': 'mesa_profile_substrings',
        'MESA default suffix': 'mesa_suffix',
        'GYRE base folder/directory': 'gyre_base_dir',
        'GYRE default output folder/directory': 'gyre_output_dir',
        'GYRE default detail substring': 'gyre_detail_substring',
        'GYRE default summary substring': 'gyre_summary_substring',
        'POLYTROPE structure base folder/directory': 'poly_base_dir',
        'POLYTROPE structure default output folder/directory': 'poly_struct_dir',
        'POLYTROPE default structure substrings': 'poly_struct_substrings',
        'POLYTROPE default structure suffix': 'poly_struct_suffix',
        'POLYTROPE mode base folder/directory': 'poly_mode_base_dir',
        'POLYTROPE mode default output folder/directory': 'poly_mode_dir',
        'POLYTROPE mode default detail substring': 'poly_mode_substrings',
        'POLYTROPE mode default summary substring': 'poly_summary_substring',
    }

    # get converted keys
    @classmethod
    def convert(cls, my_read_key: str) -> str:
        """Converts the comment read from file into the dictionary key.

        Parameters
        ----------
        my_read_key : str
            The comment read from file.

        Returns
        -------
        converted key: str
            The converted dictionary key.
        """
        if (
            converted_key := cls.CONVERSION_DICT.value.get(my_read_key)
        ) is None:
            logger.error(f'Invalid key provided. ({my_read_key}) Now exiting.')
            sys.exit()
        else:
            return converted_key


# enumeration class containing default values for the information directories and files used during initialization of the 'info_dir' reader
class EnumerationInitializationDirs(EdsR):
    """Enumeration class containing default values for the information directories and files used during initialization of the 'info_dir' reader."""

    DIRECTORY_PATH = 'data/setup_data/'
    FILE_NAME = 'information_folders.dat'


# enumeration class containing default values for the model load lists
class EnumerationLoadLists(EdsR):
    """Enumeration class containing default values for the model load lists."""

    GYRE_MODEL_LOAD_LIST = [
        'base_dir',
        'mesa_output_dir',
        'mesa_profile_substrings',
        'mesa_suffix',
        'gyre_detail_substring',
        'gyre_summary_substring',
        'gyre_output_dir',
    ]
    GYRE_MODEL_COMPARISON_LIST = [
        'base_dir',
        'mesa_output_dir',
        'mesa_profile_substrings',
        'mesa_suffix',
        'gyre_detail_substring',
        'gyre_summary_substring',
    ]
