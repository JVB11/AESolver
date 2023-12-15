"""Python module containing class that reads information about the information directories from a specific file.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import ast
import logging
import shlex
import sys
from pathlib import Path

# relative imports to handle information dictionaries
from .enumeration_files import InfoDictConv, EnumInitDir, EnumLoadLists

# custom module imports
from path_resolver import resolve_path_to_file

# type checking imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


# set up logger
logger = logging.getLogger(__name__)


# class that reads information
class InformationDirectoriesHandler:
    """Python class that handles how information directories are read from the configuration file.

    Parameters
    ----------
    alternative_directory_path : str or None, optional
        Custom path to information directory if a string. Uses default directory path ('./data/setup_data/') if None; by default None.
    alternative_file_name : str or None, optional
        Custom configuration file name if a string. Uses default configuration file name ('information_folders.dat') if None; by default None.
    toml_use : bool, optional
        Specifies if toml-style inlist was used that contains the information folder data; by default False.
    """

    # type hints for attributes
    _sys_args: list[str]
    _use_toml_inlist: bool
    _toml_inlist_path: str | None
    _directory_path: str
    _file_name: str
    _full_path: str
    _information_dict: "dict[str, Any]"
    _gyre_model_load_list: list[str]
    _gyre_model_comparison_list: list[str]
    # set up a dummy variable to denote whether pytest is calling this module
    _called_from_pytest: bool
    _pytest_file_path: None | Path

    # initialization method
    def __init__(
        self,
        sys_arguments: list[str],
        called_from_pytest: bool = False,
        pytest_file_path: None | Path = None,
        alternative_directory_path: str | None = None,
        alternative_file_name: str | None = None,
        toml_use: bool = False,
        inlist_path: str | None = None,
    ) -> None:
        # store system arguments
        self._sys_args = sys_arguments
        self._called_from_pytest = called_from_pytest
        self._pytest_file_path = pytest_file_path
        # store the called from pytest variable
        # - inlist-specific setup
        if toml_use and (inlist_path is not None):
            # toml inlist parse
            self._use_toml_inlist = True
            self._toml_inlist_path = inlist_path
        else:
            # custom data file parse
            self.non_toml_init(
                alternative_directory_path=alternative_directory_path,
                alternative_file_name=alternative_file_name,
            )
            self._use_toml_inlist = False
            self._toml_inlist_path = None
        # - generic setup
        # initialize the information dictionary
        self._information_dict = {}
        # initialize the model load lists
        self._gyre_model_load_list = EnumLoadLists.get_relevant_attr(
            attrname='gyre_model_load_list'
        )
        self._gyre_model_comparison_list = EnumLoadLists.get_relevant_attr(
            attrname='gyre_model_comparison_list'
        )

    def non_toml_init(
        self,
        alternative_directory_path: str | None,
        alternative_file_name: str | None,
    ) -> None:
        """Initializes the class without use of a toml-style inlist.

        Parameters
        ----------
        alternative_directory_path : str | None
            Custom path to information directory if a string. Uses default directory path ('./data/setup_data/') if None.
        alternative_file_name : str | None
            Custom configuration file name if a string. Uses default configuration file name ('information_folders.dat')
        """
        # store the actual paths
        self._directory_path = EnumInitDir.get_relevant_attr(
            attrname='directory_path', input_val=alternative_directory_path
        )
        self._file_name = EnumInitDir.get_relevant_attr(
            attrname='file_name', input_val=alternative_file_name
        )
        # resolve the path to the data file
        # -- ADDITIONAL PYTEST SPECIFIC COMMANDS SO THAT THE PATH TO THE INLIST MAY BE RESOLVED
        if self._called_from_pytest:
            if TYPE_CHECKING:
                assert isinstance(self._pytest_file_path, Path)
            # use the pytest path variable to resolve the path
            pytest_sys_args_dummy = [str(Path(self._pytest_file_path).parent)]
            resolved_path = resolve_path_to_file(
                sys_arguments=pytest_sys_args_dummy,
                file_name=self._file_name,
                default_path=self._directory_path,
                default_run_path='../../',
            )
        else:
            # use the regular system arguments to resolve the path
            resolved_path = resolve_path_to_file(
                sys_arguments=self._sys_args,
                file_name=self._file_name,
                default_path=self._directory_path,
                default_run_path='../../',
            )
        # check if the file is present at the location
        if resolved_path.exists():
            self._full_path = str(resolved_path)
        else:
            raise FileNotFoundError(
                f'The information directories file was not found. (path: {self._directory_path}, file: {self._file_name})'
            )

    @property
    def full_info(self) -> "dict[str, Any]":
        """Returns the full information dictionary.

        Returns
        -------
        dict[str, Any]
            Contains information about the information directories.
        """
        return self._information_dict

    @property
    def load_info(self) -> "dict[str, Any]":
        """Returns the information dictionary containing only the necessary info for loading GYRE information.

        Returns
        -------
        dict[str, Any]
            Contains the necessary information for loading GYRE files.
        """
        return self._get_info_dict('_gyre_model_load_list')

    @property
    def gyre_comparison_info(self) -> "dict[str, Any]":
        """Returns the information dictionary containing only the necessary info for loading information that performs GYRE comparisons.

        Returns
        -------
        dict[str, Any]
            Contains the necessary information for comparing GYRE outputs.
        """
        return self._get_info_dict('_gyre_model_comparison_list')

    # utility method used to retrieve specific information from lists
    def _get_info_dict(self, my_attr: str) -> "dict[str, Any]":
        """Utility method used to retrieve specific information from a list specified by its attribute name.

        Parameters
        ----------
        my_attr : str
            Specifies which information shall be retrieved.

        Returns
        -------
        dict[str, Any]
            The dictionary containing the specific information.
        """
        _attr_list = getattr(self, my_attr)
        return {
            _k: _v
            for _k, _v in self._information_dict.items()
            if _k in _attr_list
        }

    # parse custom inlist/data file
    def _parse_custom_info_files(
        self, dictionary_information: "dict[str, Any] | None"
    ) -> None:
        """Performs parsing for the custom-style information folders data files.

        Parameters
        ----------
        dictionary_information : dict[str, Any] | None
            If a dictionary, store parsed information in this dictionary. If None, create a new dictionary to store information.
        """
        # make immutable dictionary argument
        if dictionary_information is None:
            dictionary_information = {}
        # parse the information directories file
        try:
            with open(f'{self._full_path}') as _info_file:
                # read the complete information file as a string and initialize the lexical analysis engine (shlex object)
                _my_str = '\n'.join(_info_file.readlines())
                _lex = shlex.shlex(_my_str)
                _lex_comments = shlex.shlex(_my_str)
                # set the commenter symbols
                _lex.commenters = '#'
                _lex_comments.commenters = ''  # no comments here
                # add the round brackets to the word characters, in order to obtain tuple values in full.
                _lex.wordchars += ','
                _lex.wordchars += '.'
                _lex.wordchars += '('
                _lex.wordchars += ')'
                # make sure comments are picked up
                _lex_comments.wordchars += '#'
                _lex_comments.wordchars += ' '
                _lex_comments.wordchars += '/'
                # add the quotes
                _lex.quotes = '\'"'
                # add escape quotes to denote lines
                _lex.escape = '\n'
                _lex_comments.escape = '\n'
                # stop the whitespace split
                _lex_comments.whitespace = '\n'
                # analyze the commented values to extract names
                _names = [
                    InfoDictConv.convert(_x[2:])
                    for _x in _lex_comments
                    if '#' in _x
                ]
                # iterate over the analyzed strings
                for _i, _n in zip(_lex, _names):
                    try:
                        self._information_dict[_n] = ast.literal_eval(_i)
                    except SyntaxError:
                        print(
                            f'{_i} could not be parsed using our lexical analyzer. Now exiting.'
                        )
                        sys.exit()
        except NameError:
            logger.exception('A variable was/variables were not defined.')
        except TypeError:
            logger.exception('The type of a variable/variables was wrong.')

    # parse the toml-style information directories file
    def _parse_toml_info_file(self) -> None:
        """Performs parsing for the toml-style information folders data file (i.e. the toml-style inlist)."""
        # guard the tomllib import (which is python version dependent!)
        try:
            # import the toml inlist handler which contains the tomllib import
            from inlist_handler import TomlInlistHandler

            # load the full toml-style inlist
            my_toml_data = TomlInlistHandler.get_inlist_values(
                self._toml_inlist_path
            )
            # get symbolic links for sub-dictionaries
            infodict = my_toml_data['information_folders']
            MESAdict = infodict['MESA']
            GYREdict = infodict['GYRE']
            POLY_structure_dict = infodict['POLY']['structure']
            POLY_mode_dict = infodict['POLY']['modes']
            # parse the inlist for requested data
            # - MESA info
            self._information_dict['base_dir'] = MESAdict['default']['base_dir']
            self._information_dict['mesa_output_dir'] = MESAdict['default'][
                'output_dir'
            ]
            self._information_dict['mesa_profile_substrings'] = (
                tuple(msel['substrings'])
                if (msel := MESAdict['selection'])['substrings_to_tuple']
                else msel['substrings']
            )
            self._information_dict['mesa_suffix'] = MESAdict['selection'][
                'suffix'
            ]
            # - GYRE info
            self._information_dict['gyre_base_dir'] = GYREdict['default'][
                'base_dir'
            ]
            self._information_dict['gyre_output_dir'] = GYREdict['default'][
                'output_dir'
            ]
            self._information_dict['gyre_detail_substring'] = (
                tuple(gsel['detail_substrings'])
                if (gsel := GYREdict['selection'])['detail_substrings_to_tuple']
                else gsel['detail_substrings']
            )
            self._information_dict['gyre_summary_substring'] = GYREdict[
                'selection'
            ]['summary_substring']
            # - POLY structure info
            self._information_dict['poly_base_dir'] = POLY_structure_dict[
                'default'
            ]['base_dir']
            self._information_dict['poly_struct_dir'] = POLY_structure_dict[
                'default'
            ]['output_dir']
            self._information_dict['poly_struct_substrings'] = (
                tuple(psds['substrings'])
                if (psds := POLY_structure_dict['selection'])[
                    'substrings_to_tuple'
                ]
                else psds['substrings']
            )
            self._information_dict['poly_struct_suffix'] = POLY_structure_dict[
                'selection'
            ]['suffix']
            # - POLY mode info
            self._information_dict['poly_mode_base_dir'] = POLY_mode_dict[
                'default'
            ]['base_dir']
            self._information_dict['poly_mode_dir'] = POLY_mode_dict['default'][
                'output_dir'
            ]
            self._information_dict['poly_mode_substrings'] = (
                tuple(pmds['detail_substrings'])
                if (pmds := POLY_mode_dict['selection'])[
                    'detail_substrings_to_tuple'
                ]
                else pmds['detail_substrings']
            )
            self._information_dict['poly_summary_substring'] = POLY_mode_dict[
                'selection'
            ]['summary_substring']
        except ImportError:
            logger.error(
                'Cannot import the TomlInlistHandler module. Probably cannot import the tomllib module (depending on the python version). Use a custom-style inlist! Now exiting.'
            )
            sys.exit()

    # parse the information directories file
    def parse_info_directories(
        self, dictionary_information: "dict[str, Any] | None" = None
    ):
        """Parses the file containing the information directories.

        Parameters
        ----------
        dictionary_information : dict[str, Any] | None, optional
            If a dictionary, store parsed information in this dictionary. If None, create a new dictionary to store information; by default None.
        """
        if not self._use_toml_inlist:
            # custom data file parsing
            self._parse_custom_info_files(
                dictionary_information=dictionary_information
            )
        else:
            # toml-style inlist file parsing
            self._parse_toml_info_file()
