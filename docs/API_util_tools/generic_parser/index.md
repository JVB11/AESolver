---
layout: default
title: index of generic_parser module
permalink: /API_util_tools/generic_parser/index.html
---

Module generic_parser

=====================

Python package initialization file for package containing a generic class that can be subclassed to provide a parser used to parse arguments from the command line and/or an inlist.

Detailed information on the available private and publicly available methods is available in the [module API reference](API_util_tools/generic_parser/generic_parser.html).

Sub-modules
-----------

* generic_parser.generic_parser

Classes
-------

`GenericParser(base_directory_name: str, full_inlist_path: pathlib.Path | None = None, base_dir: str | None = None, inlist_name: str | None = None, inlist_dir: str | None = None, inlist_suffix: str = 'in')`
:   (Super)Class containing generic methods and arguments used for parsing of input. Input can either be read from the command-line argument parser, or from inlist files.

    ### Instance variables

    `inlist_path`
    :   Returns the path to the used/selected inlist.

    ### Methods

    `get_ha(self, ha_dict, index_val)`
    :   Hierarchical indexing utility method used to access (possibly nested) dictionary elements.

    `read_args(self)`
    :   Generic method used to read relevant arguments from the parsed information.

    `read_custom_args(self)`
    :   Method used to read relevant arguments from the information parsed from a custom inlist.

    `read_in_to_list(self, argument_list)`
    :   Generic method used to read list of arguments in to a list.

    `read_toml_args(self)`
    :   Method used to read relevant arguments from the parsed toml inlist file.
