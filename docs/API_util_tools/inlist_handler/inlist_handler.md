---
layout: api_module_page
title: InlistHandler class API reference
permalink: /overview_API/API_util_tools/inlist_handler/inlist_handler.html
---

# inlist_handler.inlist_handler module

API reference for Python module that defines the class needed to parse custom-format inlists and retrieve the necessary information for use in Python programs.

{% include button_api_module.html referenced_path="/tree/main/util-tools/inlist_handler/inlist_handler.py" %}

## Classes

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
    :   Utility method that retrieves the default inlist values, and updates them, if necessary.
        
        Parameters
        ----------
        inlist_path: str
            Name of the inlist used to define the input for the run.
        
        Returns
        -------
        dictionary_inlist: dict
            Contains the key-value pairs of the input parameters specified in the inlist.
