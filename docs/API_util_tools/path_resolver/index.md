---
layout: default
title: index for path_resolver module
permalink: /path_resolver/index.html
---

Module path_resolver

====================

Initialization file for the module containing functions used to resolve paths.

Detailed information on the available private and publicly available functions is available in the [module API reference](API_util_tools/path_resolver/resolve_paths.html).

Sub-modules
-----------

* path_resolver.resolve_paths

Publicly Available Functions
----------------------------

`resolve_path_to_file(sys_arguments: list, file_name: str, default_path: str, default_run_path: str = '..') -> pathlib.Path`
:   Resolves the path to a file with known default relative path from the base directory.
