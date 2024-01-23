"""Python module containing (super)class that handles generic parsing of arguments, and inlist reading.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import argparse
import logging
import sys
from pathlib import Path

# load custom inlist_handler module --> NOTE: required for this package
from inlist_handler import InlistHandler

# typing imports
from typing import Any


# set up logger
logger = logging.getLogger(__name__)


# class containing generic arguments and methods used for parsing
class GenericParser:
    """(Super)Class containing generic methods and arguments used for the parsing of input. Input can either be read from the command-line argument parser, or from inlist files.

    Notes
    -----
    This class/object can be initialized in two different ways:

    1. Either you provide the full (pathlib) Path to the inlist file that you want to parse from, or,
    2. you provide the strings representing the base directory, inlist name, inlist directory, and inlist suffix.

    If you do not provide the necessary information for one of these initialization methods, the initialization will fail with an ExceptionGroup that denotes the class'/object's failures in understanding the user input.

    Parameters
    ----------
    base_directory_name : str
        Name of the base directory (of your github repo in which this parser is used, for example).
    full_inlist_path : Path | None, optional
        Initialization mode 1: full (pathlib) Path to the inlist file that you want to parse; by default None (i.e. opting to perform actions in this class using initialization option 2).
    base_dir : str | None, optional
        Initialization mode 2: The base directory in which the inlists directory can be found; by default None (i.e. opting to perform actions in this class using initialization option 1).
    inlist_name : str | None, optional
        Initialization mode 2: The name of the inlist file; by default None (i.e. opting to perform actions in this class using initialization option 1).
    inlist_dir : str | None, optional
        Initialization mode 2: The name of the inlists directory in the base directory. If None, and initialization mode 1 is not used (i.e. 'full_inlist_path' is None), the parsing object looks for a directory named 'inlists/user_input/'; by default None.
    inlist_suffix : str, optional
        Initialization mode 2: The suffix of the inlist file (only used for initialization mode 2); by default 'in'.
    """

    # attribute type declarations
    _parser: argparse.ArgumentParser
    _inlist_name: str
    _base_dir: str
    _inlist_dir: str | None
    _inlist_suffix: str
    _data_read_dictionary: None | dict
    _args: argparse.Namespace

    def __init__(
        self,
        base_directory_name: str,
        full_inlist_path: Path | None = None,
        base_dir: str | None = None,
        inlist_name: str | None = None,
        inlist_dir: str | None = None,
        inlist_suffix: str = 'in',
    ) -> None:
        # set up the argument parser
        self._parser = argparse.ArgumentParser()
        # initialize the variables holding information about the inlist
        self._initialize_path_info(
            base_directory_name=base_directory_name,
            full_inlist_path=full_inlist_path,
            base_dir=base_dir,
            inlist_name=inlist_name,
            inlist_dir=inlist_dir,
            inlist_suffix=inlist_suffix,
        )
        # store a dictionary that will be used to hold inlist data (if requested),
        # or the argument-parsed data
        self._data_read_dictionary = None
        # log info
        logger.debug('Argument parser initialized.')
        # add argument parser groups and arguments
        self._set_up_generic_argparser()  # generic groups
        self._set_up_specific_argparser()  # specific groups
        # parse the arguments, setting up a namespace (unknown arguments are ignored)
        self._args, unknown_setup_args = self._parser.parse_known_args()
        logger.debug('Arguments parsed.')
        # generate the data dictionary object
        self._prepare_read_args()
        logger.debug('Ready to read arguments.')

    @property
    def inlist_path(self):
        """Returns the string representation of the path to the used/selected inlist.

        Returns
        -------
        str
            Path to the used/selected inlist.
        """
        return self._args.inlist_name

    # initialization modes
    def _initialize_path_info(
        self,
        base_directory_name: str,
        full_inlist_path: Path | None = None,
        base_dir: str | None = None,
        inlist_name: str | None = None,
        inlist_dir: str | None = None,
        inlist_suffix: str = 'in',
    ) -> None:
        """Initialization method used to store the relevant path information in this generic parser object.

        Parameters
        ----------
        base_directory_name : str
            Name of the base directory (of your github repo in which this parser is used, for example).
        full_inlist_path : Path | None, optional
            Full Path (object) to the inlist file or None; by default None.
        base_dir : str | None, optional
            The base directory in which the inlists directory can be found; by default None.
        inlist_name : str | None, optional
            The name of the inlist file; by default None.
        inlist_dir : str | None, optional
            The name of the inlists directory in the base directory; by default None. If None, and initialization mode 1 is not used (i.e. 'full_inlist_path' is None), the parsing object looks for a directory named 'inlists/user_input/'; by default None.
        inlist_suffix : str, optional
            The suffix of the inlist file; by default 'in'.

        Raises
        ------
        ExceptionGroup
            Any of the exceptions raised during the initialization methods will be stored in this exception group, which will be raised if none of the two initialization methods succeed.
        """
        # store exception list
        _exception_list = []
        # verify if either of the two initialization modes succeed
        _mode_2_succeeded = self._initialization_mode_2(
            base_dir=base_dir,
            inlist_name=inlist_name,
            inlist_dir=inlist_dir,
            inlist_suffix=inlist_suffix,
            my_exceptions=_exception_list,
        )
        _mode_1_succeeded = self._initialization_mode_1(
            full_inlist_path=full_inlist_path,
            base_directory_name=base_directory_name,
            my_exceptions=_exception_list,
        )  # because this is the last to be run, this is also the DEFAULT/PREFERRED INITIALIZATION MODE
        # if one of the two modes succeeded: path initialized!
        if (_mode_1_succeeded and not _mode_2_succeeded) | (
            _mode_2_succeeded and not _mode_1_succeeded
        ):
            logger.debug(
                'One of the initialization methods was OK. Ready to rumble!'
            )
        # if both succeed: determine what TODO
        elif _mode_1_succeeded and _mode_2_succeeded:
            logger.debug(
                'Both initialization methods succeeded, using preferred method 1.'
            )
        # if both fail: raise exception group as the code does not understand the input
        else:
            raise ExceptionGroup(
                'Both initialization methods failed', _exception_list
            )

    # initialization mode 1
    def _initialization_mode_1(
        self,
        full_inlist_path: Path | None,
        base_directory_name: str,
        my_exceptions: list[Exception],
    ) -> bool:
        """Initialization method 1 for this generic parser class. This analyzes a passed Path, in order to retrieve/extract necessary information based on this path.

        Parameters
        ----------
        full_inlist_path : Path | None
            Full Path (object) to the inlist file, or None.
        base_directory_name : str
            Name of the base directory (of your github repo in which this parser is used, for example).
        my_exceptions : list[Exception]
            List that stores any exceptions encountered during the initialization processes.

        Returns
        -------
        bool
            True if this initialization method succeeded, False otherwise.
        """
        # analyze the passed Path
        # - store common part of error message
        _common_error = f'cannot be inferred from the full inlist path {full_inlist_path} (which is likely not a Path (but needs to be in order to succeed)).'
        # - get suffix
        try:
            _inlist_suffix = full_inlist_path.name.split('.')[-1]  # type: ignore
            # - get file name
            try:
                _inlist_name = full_inlist_path.name.split('.')[0]  # type: ignore
            except AttributeError:
                my_exceptions.append(
                    TypeError(f'The inlist name {_common_error}')
                )
                return False
            # - get base dir
            try:
                # get parents
                path_parents = full_inlist_path.parents  # type: ignore
                # get the base directory path string based on its name
                for i in range(len(path_parents)):
                    if (
                        base_directory_name
                        in (parent_path := path_parents[i]).name
                    ):
                        _base_dir = str(parent_path)
                        break
                # NO BASE DIRECTORY FOUND!
                else:
                    my_exceptions.append(
                        FileNotFoundError(
                            f'The base directory name {base_directory_name} was not found in the specified full inlist path {full_inlist_path}.'
                        )
                    )
                    return False
            except AttributeError:
                my_exceptions.append(
                    TypeError(f'The base directory {_common_error}')
                )
                return False
            # - get (relative) path to inlist directory
            try:
                # get parent directory path
                full_inlist_dir_path = full_inlist_path.parent  # type: ignore
                # use that to obtain and store the string representation of the relative path to the base directory
                _inlist_dir = str(
                    full_inlist_dir_path.relative_to(Path(_base_dir))
                )
            except AttributeError:
                my_exceptions.append(
                    TypeError(f'The inlist directory {_common_error}')
                )
                return False
        except AttributeError:
            my_exceptions.append(
                TypeError(
                    f'The inlist suffix {_common_error} The other variables cannot be inferred if there is no suffix!'
                )
            )
            return False
        else:
            # now store those variables in the class member
            self._inlist_suffix = _inlist_suffix
            self._inlist_name = _inlist_name
            self._base_dir = _base_dir
            self._inlist_dir = _inlist_dir
            # return that all operation went ahead as planned
            return True

    # initialization mode 2
    def _initialization_mode_2(
        self,
        base_dir: str | None,
        inlist_name: str | None,
        inlist_dir: str | None,
        inlist_suffix: str,
        my_exceptions: list[Exception],
    ) -> bool:
        """Initialization method 2 for this generic parser class. This method uses passed information to initialize the object.

        Parameters
        ----------
        base_dir : str | None
            The base directory in which the inlists directory can be found, or None.
        inlist_name : str | None
            Name of the inlist file (without suffix and path information).
        inlist_dir : str | None
            Name of the directory containing the inlist file (i.e., string representation of the path to this directory).
        inlist_suffix : str
            Suffix of the inlist file.
        my_exceptions : list[Exception]
            List that stores any exceptions encountered during the initialization processes.

        Returns
        -------
        bool
            True if this initialization method succeeded, False otherwise.
        """
        # check if the necessary information is available (and of the right type!)
        ok1 = self._check_if_string(
            variable_description='base directory',
            my_value=base_dir,
            my_exceptions=my_exceptions,
        )
        ok2 = self._check_if_string(
            variable_description='inlist_name',
            my_value=inlist_name,
            my_exceptions=my_exceptions,
        )
        all_ok = ok1 & ok2
        # store the necessary data
        if all_ok:
            # all types OK, store necessary data
            self._inlist_name = inlist_name  # type: ignore
            self._base_dir = base_dir  # type: ignore
            # you do not check the type of inlist_dir to allow for a default to be built in!
            self._inlist_dir = (
                'inlists/user_input' if inlist_dir is None else inlist_dir
            )
            self._inlist_suffix = inlist_suffix
            return True
        # return whether every action succeeded
        else:
            return False

    @staticmethod
    def _check_if_string(
        variable_description: str, my_value: Any, my_exceptions: list[Exception]
    ) -> bool:
        """Checks if the passed value is a string, and adds an exception to the stored exception list if it is not!

        Parameters
        ----------
        variable_description : str
            Describes the use of the variable.
        my_value : Any
            The value that needs to be of string type (for this method to succeed/not add an exception to the list).
        my_exceptions : list[Exception]
            Keeps track of the raised exceptions during initialization.

        Returns
        -------
        bool
            True if the input value ('my_value') was a string, False otherwise.
        """
        if not isinstance(my_value, str):
            my_exceptions.append(
                TypeError(
                    f'Variable {variable_description} with value {my_value} is not of the string type (and should be).'
                )
            )
            # return statement
            return False
        else:
            return True

    # add generic groups and arguments
    def _set_up_generic_argparser(self) -> None:
        """Method used to set up GENERIC argument parser groups and arguments.

        Notes
        -----
        NOT meant to be overloaded by its subclass.
        """
        # set up a group that contains generic arguments
        generic_args = self._parser.add_argument_group(
            'Generic', 'Group of generic arguments.'
        )
        generic_args.add_argument(
            '-no_verbose',
            '--no_verbose',
            action='store_true',
            default=False,
            help='If True, do not print/log verbose output. '
            'If False, print/log verbose output.',
        )
        generic_args.add_argument(
            '-debug',
            '--debug',
            action='store_true',
            default=False,
            help='If True, log debug messages. '
            'If False, do not log debug messages.',
        )
        # set up a group that defines whether values are read from inlist or as
        # command line arguments
        parser_selection = self._parser.add_argument_group(
            'Parser/inlist selection',
            'Group of arguments related to using parser (command-line arguments) '
            'or inlists as input arguments.',
        )
        parser_selection.add_argument(
            '-nil',
            '--no_inlist',
            action='store_true',
            default=False,
            help='If True (default=False), use specified command-line arguments '
            'to parse commands, otherwise, use an inlist.',
        )
        # try adding information on the base inlist
        # - set up inlist name variable
        try:
            _my_default_inlist_path = self._exists(
                f'{self._base_dir}/{self._inlist_dir}/{self._inlist_name}.{self._inlist_suffix}'
            )
        except TypeError:
            # log information that faulty default path was supplied
            logger.exception(
                f'Faulty path to base directory ({self._base_dir}), '
                f'inlist directory ({self._inlist_dir}), and/or inlist file name '
                f'({self._inlist_name}.{self._inlist_suffix}) was supplied. Now exiting.'
            )
            sys.exit()
        # - add argument
        parser_selection.add_argument(
            '--inlist_name',
            type=str,
            default=_my_default_inlist_path,
            help='Specify the name of the inlist.',
        )

    # check if file or directory exists, raise error if not
    def _exists(self, my_path, is_file=True):
        """Generic method used to check if a file exists.

        Parameters
        ----------
        my_path : str
            The (string representation of the) path to the file or directory.
        is_file : bool, optional
            If True, check a path to a file. If False, check a path to directory; by default True.

        Returns
        -------
        my_path : str
            The valid path. Will not be returned if the path is not valid; in that case a TypeError shall be raised.
        """
        # generate the path object
        _my_path_object = Path(my_path)
        # check the validity of the path
        _valid = (is_file and _my_path_object.is_file()) or (
            (not is_file) and _my_path_object.is_dir()
        )
        # return the path if valid, or raise type error
        if _valid:
            return my_path
        else:
            raise TypeError

    # prepare to read arguments: read inlist if necessary
    def _prepare_read_args(self) -> None:
        """Internal method used to prepare to read arguments: reads the inlist if necessary, and otherwise generates the data dictionary from passed command-line arguments."""
        # using command-line arguments
        if self._args.no_inlist:
            self._data_read_dictionary = vars(self._args)
        # using inlist arguments
        else:
            # toml inlist
            if self._inlist_suffix == 'toml':
                try:
                    from inlist_handler import TomlInlistHandler

                    self._data_read_dictionary = (
                        TomlInlistHandler.get_inlist_values(
                            self._args.inlist_name
                        )
                    )
                except ImportError:
                    logger.error(
                        'Cannot load the tomllib module, which is probably because of a wrong python version being used. Use custom-style inlists instead! Now exiting.'
                    )
                    sys.exit()
            # custom inlist
            else:
                self._data_read_dictionary = InlistHandler.get_inlist_values(
                    self._args.inlist_name
                )

    # generic method used to read list of tuples of arguments into dictionary
    def read_in_to_dict(self, tuple_list: list[tuple]) -> dict:
        """Generic method used to read list of tuples of arguments into a dictionary.

        Notes
        -----
        If a(n optional) third value is supplied in the tuple list, it should be provided for all input arguments. Use False for the negating bool variable (optional third entry) when entering non-boolean input!

        Parameters
        ----------
        tuple_list : list[tuple]
            List containing tuples that represent elements of the dictionary: first element is the dictionary key. The second element is the dictionary value. The optional third element is a boolean that inverts a boolean dictionary value if True. It should be set to False (when specified) for non-boolean dictionary values.

        Returns
        -------
        my_data_dict : dict
            The dictionary containing the loaded items.
        """
        # generate the empty dictionary
        my_data_dict = {}
        # fill the dictionary depending on input
        try:
            for _my_key, _my_value, _my_negating_bool in tuple_list:
                _val = self.get_ha(self._data_read_dictionary, _my_value)
                my_data_dict[_my_key] = not _val if _my_negating_bool else _val
        except ValueError:
            for _my_key, _my_value in tuple_list:
                my_data_dict[_my_key] = self.get_ha(
                    self._data_read_dictionary, _my_value
                )
        # return the dictionary
        return my_data_dict

    # generic method used to read list of arguments into list
    def read_in_to_list(self, argument_list):
        """Generic method used to read list of arguments into a list.

        Notes
        -----
        If a(n optional) second value is supplied in the argument (tuple) list, it should be provided for all input arguments. Use False for the negating bool variable (optional second entry) when entering non-boolean input!

        Parameters
        ----------
        argument_list : list[str] | list[tuple]
            List containing arguments that represent the names of the arguments read in the inlist or argumentparser. Optional second (tuple) value is a boolean that inverses a boolean value if True.

        Returns
        -------
        list
            The list of data containing the requested parsed arguments values.
        """
        # fill and return the list
        try:
            return [
                not self.get_ha(self._data_read_dictionary, _my_val)
                if _negating_bool
                else self.get_ha(self._data_read_dictionary, _my_val)
                for _my_val, _negating_bool in argument_list
            ]
        except ValueError:
            return [
                self.get_ha(self._data_read_dictionary, _my_val)
                for _my_val in argument_list
            ]

    # access elements in (possibly nested) dicts
    def get_ha(self, ha_dict: dict, index_val: list[str]) -> int | str | float | list:
        """Hierarchical indexing utility method used to access (possibly nested) dictionary elements.

        Parameters
        ----------
        ha_dict : dict
            (Possibly nested) dictionary containing input elements.
        index_val : list[str]
            Index list for the (possibly nested) dict.

        Returns
        -------
        int | str | float | list
            Requested value.
        """
        try:
            # key list
            if (first_key := index_val[0]) not in ha_dict:
                # element not present, return None
                return None
            elif len(index_val) == 1:
                # return the element given for the key in the dict
                return ha_dict[first_key]
            else:
                # recursively retrieve deeper elements in the dict
                return self.get_ha(ha_dict[first_key], index_val[1:])
        except IndexError:
            # key string
            return ha_dict[index_val]

    # generic method used to read input arguments
    def read_args(self):
        """Generic method used to read relevant arguments from the parsed information. The output of this method is dependent on the overloading of the read_toml_args or read_custom_args methods."""
        # read action selected based on inlist suffix
        if self._inlist_suffix == 'toml':
            return self.read_toml_args()
        else:
            return self.read_custom_args()

    # add argument parser groups and arguments
    def _set_up_specific_argparser(self) -> None:
        """Method used to set up the SPECIFIC argument parser groups and arguments.

        Notes
        -----
        Meant to be overloaded by its subclass.
        """
        pass

    # read custom inlist input arguments
    def read_custom_args(self):
        """Method used to read relevant arguments based on the information parsed from a custom inlist file.

        Notes
        -----
        Meant to be overloaded by its subclass.
        """
        pass

    # read toml input arguments
    def read_toml_args(self):
        """Method used to read relevant arguments from the parsed toml inlist file.

        Notes
        -----
        Meant to be overloaded by its subclass.
        """
        pass
