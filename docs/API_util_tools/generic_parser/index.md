---
layout: default
title: index of generic_parser module
permalink: /overview_API/API_util_tools/generic_parser/index.html
---

# Index for generic_parser module

Index for the generic_parser package containing a generic class that can be subclassed to provide a parser used to parse arguments from the command line and/or an inlist.

Detailed information on the available private and publicly available methods is available in the [module API reference](generic_parser.html).
In this index, the publicly available methods and properties of the parser class are listed (with a short description of their functionality).

## Sub-modules

* generic_parser.generic_parser

## Classes

`GenericParser(base_directory_name: str, full_inlist_path: pathlib.Path | None = None, base_dir: str | None = None, inlist_name: str | None = None, inlist_dir: str | None = None, inlist_suffix: str = 'in')`
:   (Super)Class containing generic methods and arguments used for the parsing of input. Input can either be read from the command-line argument parser, or from inlist files.

    ### Instance variables

    `inlist_path`
    :   Returns the string representation of the path to the used/selected inlist.

    ### Methods

    `get_ha(self, ha_dict: dict, index_val: list[str]) -> int | str | float | list`
    :   Hierarchical indexing utility method used to access (possibly nested) dictionary elements. 'ha_dict' is the possible nested dictionary containing input elements, and 'index_val' is the index list for the possible nested dictionary.

    `read_args(self)`
    :   Generic method used to read relevant arguments from the parsed information.

    `read_custom_args(self)`
    :   Method used to read relevant arguments based on the information parsed from a custom inlist file.

    `read_in_to_list(self, argument_list: list[str] | list[tuple]) -> list`
    :   Generic method used to read a list of arguments into a list. 'argument_list' is the list containing arguments that represent the names of arguments to be read in the inlist file or argumentparser.

    `read_in_to_dict(self, tuple_list: list[tuple]) -> dict`
    :   Generic method used to read list of tuples of arguments into a dictionary. 'tuple_list' is the list containing tuples that represent elements of the dictionary (see API for more detailed description of the tuples in this list). 

    `read_toml_args(self)`
    :   Method used to read relevant arguments from the parsed toml inlist file.
