---
layout: api_module_page
title: TomlInlistHandler class API reference
permalink: /overview_API/API_util_tools/inlist_handler/toml_inlist_handler.html
---

# inlist_handler.toml_inlist_handler module

API reference for Python module that defines the class needed to parse toml-format inlists and retrieves the necessary information.

{% include button_api_module.html referenced_path="/tree/main/util-tools/inlist_handler/toml_inlist_handler.py" %}

## Classes

`TomlInlistHandler()`
:   Python class that handles how toml-format inlists are handled.

    ### Public class methods

    `get_inlist_values(cls, inlist_path: str) -> dict`
    :   Utility method that retrieves the inlist values, as parsed from the toml inlist file.
        ~~~        
        Parameters
        ----------
        inlist_path : str
            Path to the toml inlist file.
        
        Returns
        -------
        toml_input_data : dict
            Contains the key-value pairs of the input parameters specified in the inlist.
        ~~~

    ### Private class methods
    
    `_parse_toml_inlist(cls, inlist: str) -> dict`
    :   Utility method that parses the toml-format inlist.
        ~~~
        Parameters
        ----------
        inlist_path : str
            Path to the toml-format inlist file.

        Returns
        -------
        parsed_dictionary : dict
            Contains the parsed key-value pairs.
        ~~~
    
    `_adjust_for_none(cls, parsed_dict: dict) -> None`
    :   Adjust parsed dictionary values for None-value input.
        ~~~
        Parameters
        ----------
        parsed_dict : dict
            Contains the parsed dictionary keys and values.
        ~~~

{% include button_back.html %}
