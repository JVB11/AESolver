"""Python module that defines the class needed to parse custom-style inlists and retrieve the necessary information for use in Python programs.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import the necessary packages
import ast
import logging
import re
import shlex
import sys
from functools import partialmethod

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


# set up logger
logger = logging.getLogger(__name__)


# inlist handler class
class InlistHandler:
    """Python class that handles how custom-format inlists are parsed."""

    # --------------------------------------------------------------- #
    # Define the method that will be used for functional type mapping #
    # --------------------------------------------------------------- #
    @staticmethod
    def _multi_check_method(
        value_string: str, check_values: list[str], any_check: bool=False, all_check: bool=False
    ) -> bool:
        """Internal utility method used to check a condition / multiple conditions for the typing.

        Parameters
        ----------
        value_string: str
            The input string whose type needs to be verified.
        check_values: list[str]
            Specific values of the string that need to be checked.
        any_check: bool, optional
            If True, perform a 'any' check. If False, do not perform a 'any' check; by default False.
        all_check: bool, optional
            If True, perform a 'all' check. If False, do not perform a 'all' check; by default False.

        Returns
        -------
        bool
            The outcome of the check for the typing.
        """
        # return the outcome of the check
        if any_check:
            # check if any of the conditions match
            return any(
                _check_value in value_string for _check_value in check_values
            )
        elif all_check:
            # check if all conditions match
            return all(
                _check_value in value_string for _check_value in check_values
            )
        else:
            logger.error('No specified check method mentioned. Now exiting.')
            sys.exit()

    # ---------------------------------------------------------- #
    # Define the enumeration elements used for typing operations #
    # ---------------------------------------------------------- #
    # define the functional mappings for the typing operations
    string_check = partialmethod(
        _multi_check_method, check_values=[r"'", r'"'], any_check=True
    )
    bool_check = partialmethod(
        _multi_check_method,
        check_values=['False', 'FALSE', 'True', 'TRUE'],
        any_check=True,
    )
    bool_false_check = partialmethod(
        _multi_check_method, check_values=['False', 'FALSE'], any_check=True
    )
    list_check = partialmethod(
        _multi_check_method, check_values=['[', ']'], all_check=True
    )
    tuple_check = partialmethod(
        _multi_check_method, check_values=['(', ')'], all_check=True
    )
    none_check = partialmethod(
        _multi_check_method, check_values=['None'], all_check=True
    )
    float_check = partialmethod(
        _multi_check_method, check_values=['.'], all_check=True
    )

    # ---------------------------------------------------------------------- #
    # Define the enumeration elements that allow the selection of the inlist #
    # ---------------------------------------------------------------------- #
    # define and compile the regular expression used to select the defaults inlist
    compiled_regex_defaults = re.compile(r'(.*)\/.*\/(.*)\..*')
    # define and compile the regular expression used to parse floats
    compiled_float_regex = re.compile(r'\d+[eE]-*\d+')

    # ---------------------------------------------------------- #
    # Define the method that performs the typing of parsed input #
    # ---------------------------------------------------------- #
    @classmethod
    def _typer(cls, value_string):
        """Internal utility method used to define infer types from the input in the inlists.

        Parameters
        ----------
        value_string : str
            The value read from the inlist, in string format.

        Returns
        -------
        typed_value : Any
            The value_string converted to the appropriate type.
        """
        # check for common symbols that indicate the presence of string input
        if cls.__dict__['string_check'].__get__(cls)(value_string):
            if cls.__dict__['list_check'].__get__(cls)(value_string):
                logger.debug(f'{value_string} is considered a list of strings.')
                typed_value = ast.literal_eval(value_string)
            else:
                _de_quoted_string = value_string.strip('"').strip("'")
                logger.debug(
                    f'{value_string} is considered a string, so is replaced '
                    f'with {_de_quoted_string}.'
                )
                typed_value = str(_de_quoted_string)
        # check for input that is reminiscent of a boolean value
        elif cls.__dict__['bool_check'].__get__(cls)(value_string):
            logger.debug(f'{value_string} is considered a boolean.')
            if cls.__dict__['bool_false_check'].__get__(cls)(value_string):
                typed_value = False
            else:
                typed_value = True
        # check for input that is None
        elif cls.__dict__['none_check'].__get__(cls)(value_string):
            logger.debug(f'{value_string} is considered a boolean.')
            typed_value = None
        # check for a list input
        elif cls.__dict__['list_check'].__get__(cls)(value_string):
            logger.debug(f'{value_string} is considered a list.')
            typed_value = ast.literal_eval(value_string)
        # check for a tuple input
        elif cls.__dict__['tuple_check'].__get__(cls)(value_string):
            logger.debug(f'{value_string} is considered a tuple.')
            typed_value = ast.literal_eval(value_string)
        # check for a decimal point in the input
        elif cls.__dict__['float_check'].__get__(cls)(value_string) or (
            cls.compiled_float_regex.match(value_string)
        ):
            logger.debug(f'{value_string} is considered a float.')
            typed_value = ast.literal_eval(value_string)
        # otherwise, the value is treated as an integer
        else:
            logger.debug(f'{value_string} is considered an integer.')
            typed_value = int(value_string)
        return typed_value

    # ------------------------------------------- #
    # Define the method that performs the parsing #
    # ------------------------------------------- #
    @classmethod
    def _parse_inlist(cls, inlist_path, dictionary_inlist=None):
        """Internal utility method that parses a user inlist and obtains the values of the input parameters.

        Parameters
        ----------
        inlist_path: str
            Name of the inlist used for a run.
        dictionary_inlist: dict | None, optional
            Will contain (updated) key-value pairs of the values specified in the inlist. If None, no key-value pairs are specified; by default None.

        Returns
        -------
        dictionary_inlist: dict
            Contains the (updated) key-value pairs of the values specified in the inlist.
        """
        # make immutable argument
        if dictionary_inlist is None:
            dictionary_inlist = {}
        # parse the inlist
        try:
            with open(inlist_path) as _inlist:
                # read the complete inlist as one string and
                # initialize the lexical analysis engine (shlex object)
                _lex = shlex.shlex('\n'.join(_inlist.readlines()))
                # replace the comment symbol by %
                _lex.commenters = '%'
                # add the decimal dot to the word characters,
                # in order to obtain float values in full.
                _lex.wordchars += '.'
                # add the minus sign to the word characters,
                # in order to obtain negative values in full.
                _lex.wordchars += '-'
                # add the square brackets and the comma to the word characters,
                # in order to obtain list values in full.
                _lex.wordchars += '['
                _lex.wordchars += ']'
                _lex.wordchars += ','
                # add the round brackets to the word characters,
                # in order to obtain tuple values in full.
                _lex.wordchars += '('
                _lex.wordchars += ')'
                # add the quotes " and '
                _lex.quotes = '\'"'
                # add escape quotes to denote lines
                _lex.escape = '\n'
                # iterate through the analyzed strings
                for _i in _lex:
                    _keyword = _i  # get the keyword for the dictionary
                    _lex.get_token()  # remove the "=" sign
                    _value = cls._typer(_lex.get_token())
                    # get the value for the dictionary
                    # add or update the value to the dictionary
                    dictionary_inlist[f'{_keyword}'] = _value
        except NameError:
            logger.exception('A variable was/variables were not defined.')
        except TypeError:
            logger.exception('The type of a variable/variables was wrong.')
        else:
            return dictionary_inlist

    # ------------------------------------------------------------------------------- #
    # Define the method that shall be used to parse and set the default inlist values #
    # ------------------------------------------------------------------------------- #
    @classmethod
    def _get_default_inlist_values(cls, inlist_path):
        """Class method that obtains the default values obtained from the inlist: 'xxxx.defaults'.

        Parameters
        ----------
        inlist_path: str
            The name of the inlist from which the .defaults inlist name will be reconstructed.

        Raises
        ------
        NameError
            If no filepath was obtained that matches the compiled regular expression conventional naming.

        Returns
        -------
        dict
            Contains the key-value pairs of the values specified in the inlist containing the defaults.
        """
        # obtain the matches for creating the path to the defaults file
        _defaults_matches = cls.compiled_regex_defaults.match(inlist_path)
        # construct the name of the defaults file and return
        # the values of the parameters in the defaults file
        if _defaults_matches is None:
            raise NameError(
                f'No file was found at the path {inlist_path} that matches the conventional naming.'
            )
        else:
            return cls._parse_inlist(
                f'{_defaults_matches.group(1)}/defaults/'
                f'{_defaults_matches.group(2)}.defaults'
            )

    # -------------------------------------------------------- #
    # Define the (main) method that retrieves the inlist input #
    # -------------------------------------------------------- #
    @classmethod
    def get_inlist_values(cls, inlist_path: str) -> dict:
        """Utility method that retrieves the default inlist values, and updates them, if necessary.

        Parameters
        ----------
        inlist_path: str
            Name of the inlist used to define the input for the run.

        Returns
        -------
        dictionary_inlist: dict
            Contains the key-value pairs of the input parameters specified in the inlist.
        """
        # get the default values of the inlist
        _dictionary_inlist_default = cls._get_default_inlist_values(
            inlist_path=inlist_path
        )
        # get updated dictionary containing key-value pairs in the inlist
        try:
            dictionary_inlist = cls._parse_inlist(
                inlist_path, dictionary_inlist=_dictionary_inlist_default
            )
        except NameError:
            logger.exception('A variable was/variables were not defined.')
        else:
            return dictionary_inlist
