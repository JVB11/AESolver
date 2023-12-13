---
layout: default
title: index of inlist_handler API documentation
permalink: /apidoc-util-tools/inlist_handler/
---

Module inlist_handler
=====================
Initialization file for the python module that contains a class used for loading data from inlists into Python.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>

Sub-modules
-----------
* inlist_handler.inlist_handler
* inlist_handler.toml_inlist_handler

Classes
-------

`InlistHandler()`
:   Python class that handles how inlists are parsed.

    ### Class variables

    `bool_check`
    :

    `bool_false_check`
    :

    `compiled_float_regex`
    :

    `compiled_regex_defaults`
    :

    `float_check`
    :

    `list_check`
    :

    `none_check`
    :

    `string_check`
    :

    `tuple_check`
    :

    ### Static methods

    `get_inlist_values(inlist_path)`
    :   Utility method that retrieves the default inlist values, and updates them,
        if necessary.
        
        Parameters
        ----------
        inlist_path: str
            Name of the inlist used to define the input for the run.
        
        Returns
        -------
        dictionary_inlist: dict
            Contains the key-value pairs of the input parameters specified in the inlist.

`TomlInlistHandler()`
:   Python class that handles how toml-format inlists are handled.

    ### Static methods

    `get_inlist_values(inlist_path)`
    :   Utility method that retrieves the inlist values, as parsed
        from the toml inlist file.
        
        Parameters
        ----------
        inlist_path : str
            Path to the toml inlist file.
        
        Returns
        -------
        toml_input_data : dict
            Contains the key-value pairs of the input parameters
            specified in the inlist.