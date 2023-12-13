---
layout: default
title: index for path_resolver
permalink: /apidoc-util-tools/index_path_resolver/
---

Module path_resolver
====================
Initialization file for the module containing functions used to resolve paths.

Author: Jordan Van Beeck <jordan.vanbeeck@hotmail.com>

Sub-modules
-----------
* path_resolver.resolve_paths

Functions
---------

    
`resolve_path_to_file(sys_arguments: list, file_name: str, default_path: str, default_run_path: str = '..') -> pathlib.Path`
:   Resolves the path to a file with known default relative path from the base directory.
    
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
        The default relative path to the base directory, which is used to ultimately resolve the path from the current work directory.
    default_run_path : str, optional
        Conversion from the run directory to the base directory.
    
    Returns
    -------
    Path
        Resolved path to the file that you wanted to obtain.
    
    Raises
    ------
    FileNotFoundError
        When the corresponding file cannot be found in the programmatic paths. This is likely due to mis-specification of the 'default_path'!