---
layout: default
title: index of inlist_handler module
permalink: /overview_API/API_util_tools/inlist_handler/index.html
---

# Index for inlist_handler module

Initialization file for the python module that contains classes used to load data from inlists into Python.

Detailed information on the available private and publicly available functions is available in the API reference of both classes:

1.[API reference of inlist_handler](inlist_handler.html)
2.[API reference of toml_inlist_handler](toml_inlist_handler.html)

## Sub-modules

* inlist_handler.inlist_handler
* inlist_handler.toml_inlist_handler

## Classes

`InlistHandler()`
:   Python class that handles how custom-format inlists are parsed.

    ### Static methods

    `get_inlist_values(inlist_path)`
    :   Utility method that retrieves the default inlist values, and updates them, if necessary.

`TomlInlistHandler()`
:   Python class that handles how toml-format inlists are parsed.

    ### Static methods

    `get_inlist_values(inlist_path)`
    :   Utility method that retrieves the inlist values, as parsed from the toml inlist file.
