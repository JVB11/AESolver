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
    :   functional mapping that checks whether a parsed string variable is a boolean.

    `bool_false_check`
    :   functional mapping that checks whether a parsed string variable is a False (boolean) value.

    `compiled_float_regex`
    :   compiled regular expression ('re') used to parse floats.

    `compiled_regex_defaults`
    :   compiled regular expression ('re') used to select the defaults inlist.

    `float_check`
    :   functional mapping that checks whether a parsed string variable is a float.

    `list_check`
    :   functional mapping that checks whether a parsed string variable is a list.

    `none_check`
    :   functional mapping that checks whether a parsed string variable is a None value.

    `string_check`
    :   functional mapping that checks whether a parsed string variable is a string.

    `tuple_check`
    :   functional mapping that checks whether a parsed string variable is a tuple.

    ### Public class methods

    `get_inlist_values(inlist_path: str) -> dict`
    :   Utility method that retrieves the default inlist values, and updates them, if necessary.
        ~~~
        Parameters
        ----------
        inlist_path: str
            Name of the inlist used to define the input for the run.
        
        Returns
        -------
        dictionary_inlist: dict
            Contains the key-value pairs of the input parameters specified in the inlist.
        ~~~

    ### Private class methods
    
    `_get_default_inlist_values(cls, inlist_path: str) -> dict`
    :   Class method that obtains the default values obtained from the inlist: 'xxxx.defaults'.

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
    
    `_parse_inlist(cls, inlist_path: str, dictionary_inlist: dict | None=None) -> dict`
    :   Internal utility method that parses a user inlist and obtains the values of the input parameters.

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

    `_typer(cls, value_string: str) -> typing.Any`
    :   Internal utility method used to define infer types from the input in the inlists.

        Parameters
        ----------
        value_string : str
            The value read from the inlist, in string format.

        Returns
        -------
        typed_value : Any
            The value_string converted to the appropriate type.

    ### Private static methods

    `_multi_check_method(value_string: str, check_values: list[str], any_check: bool=False, all_check: bool=False) -> bool`
    :   Internal utility method used to check a condition / multiple conditions for the typing.

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

{% include button_back.html %}
