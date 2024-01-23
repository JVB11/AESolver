---
layout: api_module_page
title: path-resolving functions API reference
permalink: /overview_API/API_util_tools/path_resolver/resolve_paths.html
---

# path_resolver.resolve_paths module

Utility module containing functions used to resolve paths to files.

{% include button_api_module.html referenced_path="/tree/main/util-tools/path_resolver/resolve_paths.py" %}

## Public functions

`resolve_path_to_file(sys_arguments: list, file_name: str, default_path: str, default_run_path: str = '..') -> pathlib.Path`
:   Resolves the path to a file with a known default relative path from the base directory.
    ~~~
    Notes
    -----
    Used to hardcode relative paths in Python packages/modules.

    Parameters
    ----------
    sys_arguments : sys.argv
        The input arguments used when running the script.
    file_name : str
        Name of the file for which you would like to obtain the resolved path.
    default_path : str
        The default relative path to the base directory, which is used to ultimately resolve
        the path from the current work directory.
    default_run_path : str, optional
        Conversion from the run directory to the base directory; by default '..'.
    
    Returns
    -------
    pathlib.Path
        Resolved path to the file that you wanted to obtain.
    
    Raises
    ------
    FileNotFoundError
        Raised when the corresponding file cannot be found in the programmatic paths.
        This is likely due to mis-specification of the 'default_path'!
    ~~~

## Private functions

`_get_abspath_to_run(sys_arguments: list) -> str`
:   Get the absolute path to the run file directory.
    ~~~
    Parameters
    ----------
    sys_arguments : list
        System arguments list.

    Returns
    -------
    str
        Absolute path to the run file directory.
    ~~~

{% include button_back.html %}
