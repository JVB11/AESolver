"""Python module that defines the class needed to parse toml-style inlists and retrieves the necessary information.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
import tomllib
import logging
from collections import abc


logger = logging.getLogger(__name__)


class TomlInlistHandler:
    """Python class that handles how toml-format inlists are handled."""

    @classmethod
    def _parse_toml_inlist(cls, inlist_path: str) -> dict:
        """Utility method that parses the toml-format inlist.

        Parameters
        ----------
        inlist_path : str
            Path to the toml-format inlist file.

        Returns
        -------
        parsed_dictionary : dict
            Contains the parsed key-value pairs.
        """
        with open(inlist_path, 'rb') as fp:
            parsed_dictionary = tomllib.load(fp)
        return parsed_dictionary

    @classmethod
    def _adjust_for_none(cls, parsed_dict: dict) -> None:
        """Adjust parsed dictionary values for None-value input.

        Parameters
        ----------
        parsed_dict : dict
            Contains the parsed dictionary keys and values.
        """
        for key, val in parsed_dict.items():
            if val == {}:
                parsed_dict[key] = None
            # nested dictionary detected: parse deeper-lying values
            elif isinstance(val, abc.Mapping):
                cls._adjust_for_none(val)

    @classmethod
    def get_inlist_values(cls, inlist_path: str) -> dict:
        """Utility method that retrieves the inlist values, as parsed from the toml inlist file.

        Parameters
        ----------
        inlist_path : str
            Path to the toml inlist file.

        Returns
        -------
        toml_input_data : dict
            Contains the key-value pairs of the input parameters specified in the inlist.
        """
        toml_input_data = cls._parse_toml_inlist(inlist_path=inlist_path)
        cls._adjust_for_none(toml_input_data)
        return toml_input_data
