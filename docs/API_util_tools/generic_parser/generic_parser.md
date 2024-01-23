---
layout: api_module_page
title: GenericParser class API reference
permalink: /overview_API/API_util_tools/generic_parser/generic_parser.html
---

# generic_parser.generic_parser module

Python module containing (super)class that handles generic parsing of arguments, and inlist reading.

{% include button_api_module.html referenced_path="/tree/main/util-tools/generic_parser/generic_parser.py" %}

## Classes

`GenericParser(base_directory_name: str, full_inlist_path: pathlib.Path | None = None, base_dir: str | None = None, inlist_name: str | None = None, inlist_dir: str | None = None, inlist_suffix: str = 'in')`
:   (Super)Class containing generic methods and arguments used for the parsing of input. Input can either be read from the command-line argument parser, or from inlist files.
    ~~~
    Notes
    -----
    This class/object can be initialized in two different ways:

    1. Either you provide the full (pathlib) Path to the inlist file that you want to parse from, or,
    2. you provide the strings representing the base directory, inlist name, inlist directory, and inlist suffix.

    If you do not provide the necessary information for one of these initialization methods,
    the initialization will fail with an ExceptionGroup that denotes the class'/object's failures in 
    understanding the user input.
    
    Parameters
    ----------
    base_directory_name : str
        Name of the base directory (of your github repo in which this parser is used, for example).
    full_inlist_path : pathlib.Path | None, optional
        Initialization mode 1: full (pathlib) Path to the inlist file that you want to parse;
        by default None (i.e. opting to perform actions in this class using initialization option 2).
    base_dir : str | None, optional
        Initialization mode 2: The base directory in which the inlists directory can be found;
        by default None (i.e. opting to perform actions in this class using initialization option 1).
    inlist_name : str | None, optional
        Initialization mode 2: The name of the inlist file; by default None (i.e. opting to perform
        actions in this class using initialization option 1).
    inlist_dir : str or None, optional
        Initialization mode 2: The name of the inlists directory in the base directory. If None, and 
        initialization mode 1 is not used (i.e. 'full_inlist_path' is None), the parsing object looks
        for a directory named 'inlists/user_input/'; by default None.
    inlist_suffix : str, optional
        Initialization mode 2: The suffix of the inlist file (only used for initialization mode 2);
        by default 'in'.
    ~~~

    ### Instance variables

    `inlist_path`
    :   Returns the string representation of the path to the used/selected inlist.
        ~~~
        Returns
        -------
        str
            Path to the used/selected inlist.
        ~~~

    ### Public Methods

    `get_ha(self, ha_dict: dict, index_val: list[str]) -> int | str | float | list`
    :   Hierarchical indexing utility method used to access (possibly nested) dictionary elements.
        ~~~        
        Parameters
        ----------
        ha_dict : dict
            (Possibly nested) dictionary containing input elements.
        index_val : list[str]
            Index list for the (possibly nested) dict.
        
        Returns
        -------
        int | str | float | list
            Requested value.
        ~~~

    `read_args(self)`
    :   Generic method used to read relevant arguments from the parsed information. The output of this method is dependent on the overloading of the read_toml_args or read_custom_args methods.

    `read_custom_args(self)`
    :   Method used to read relevant arguments based on the information parsed from a custom inlist file.
        ~~~        
        Notes
        -----
        Meant to be overloaded by its subclass.
        ~~~

    `read_in_to_dict(self, tuple_list: list[tuple]) -> dict`
    :   Generic method used to read list of tuples of arguments into a dictionary.
        ~~~        
        Notes
        -----
        If a(n optional) third value is supplied in the tuple list, it should be provided for all input 
        arguments. Use False for the negating bool variable (optional third entry) when entering
        non-boolean input!
        
        Parameters
        ----------
        tuple_list : list[tuple]
            List containing tuples that represent elements of the dictionary: the first element is the 
            dictionary key. The second element is the dictionary value. The optional third element is
            a boolean that inverts a boolean dictionary value if True. It should be set to False
            (when specified) for non-boolean dictionary values.
        
        Returns
        -------
        my_data_dict : dict
            The dictionary containing the loaded items.
        ~~~

    `read_in_to_list(self, argument_list: list[str] | list[tuple]) -> list`
    :   Generic method used to read list of arguments into a list.
        ~~~        
        Notes
        -----
        If a(n optional) second value is supplied in the argument (tuple) list, it should be provided for all 
        input arguments. Use False for the negating bool variable (optional second entry) when entering 
        non-boolean input!
        
        Parameters
        ----------
        argument_list : list[str] | list[tuple]
            List containing arguments that represent the names of the arguments read in the inlist or 
            argumentparser. Optional second (tuple) value is a boolean that inverses a boolean value if True.
        
        Returns
        -------
        list
            The list of data containing the requested parsed arguments values.
        ~~~

    `read_toml_args(self)`
    :   Method used to read relevant arguments from the parsed toml inlist file.
        ~~~        
        Notes
        -----
        Meant to be overloaded by its subclass.
        ~~~

    ### Private Methods

    `_set_up_specific_argparser(self) -> None`
    :   Method used to set up the SPECIFIC argument parser groups and arguments.
        ~~~
        Notes
        -----
        Meant to be overloaded by its subclass.
        ~~~

    `_prepare_read_args(self) -> None`
    :   Internal method used to prepare to read arguments: reads the inlist if necessary, and otherwise generates the data dictionary from passed command-line arguments.

    `_exists(self, my_path: str, is_file: bool=True) -> str`
    :   Generic method used to check if a file exists.
        ~~~
        Parameters
        ----------
        my_path : str
            The (string representation of the) path to the file or directory.
        is_file : bool, optional
            If True, check a path to a file. If False, check a path to a directory; by default True.
        
        Returns
        -------
        my_path : str
            The valid path. Will not be returned if the path is not valid; in that case
            a TypeError shall be raised.
        ~~~
    
    `_set_up_generic_argparser(self) -> None`
    :   Method used to set up GENERIC argument parser groups and arguments.
        ~~~
        Notes
        -----
        NOT meant to be overloaded by its subclass.
        ~~~

    `_check_if_string(variable_description: str, my_value: typing.Any, my_exceptions: list[Exception]) -> bool`
    :   Checks if the passed value is a string, and adds an exception to the stored exception list if it is not.
        ~~~
        Parameters
        ----------
        variable_description : str
            Describes the use of the variable.
        my_value : typing.Any
            The value that needs to be of string type (for this method to succeed/not add an exception
            to the list).
        my_exceptions: list[Exception]
            Keeps track of the raised exceptions during initialization.

        Returns
        -------
        bool
            True if the input value ('my_value') was a string, False otherwise.
        ~~~

    `_initialization_mode_2(self, base_dir: str | None, inlist_name: str | None, inlist_dir: str | None, inlist_suffix: str, my_exceptions: list[Exception]) -> bool`
    :   Initialization method 2 for this generic parser class. This method uses passed information to initialize the object.
        ~~~
        Parameters
        ----------
        base_dir : str | None
            The base directory in which the inlists directory can be found, or None.
        inlist_name : str | None
            Name of the inlist file (without suffix and path information).
        inlist_dir : str | None
            Name of the directory containing the inlist file (i.e., string representation of the path
            to this directory).
        inlist_suffix : str
            Suffix of the inlist file.
        my_exceptions : list[Exception]
            List that stores any exceptions encountered during the initialization processes.
        
        Returns
        -------
        bool
            True if this initialization method succeeded, False otherwise.
        ~~~

    `_initialization_mode_1(self, full_inlist_path: pathlib.Path | None, base_directory_name: str, my_exceptions: list[Exception]) -> bool`
    :   Initialization method 1 for this generic parser class. This analyzes a passed Path, in order to retrieve/extract necessary information based on this path.
        ~~~
        Parameters
        ----------
        full_inlist_path : pathlib.Path | None
            Full Path (object) to the inlist file, or None.
        base_directory_name : str
            Name of the base directory (of your github repo in which this parser is used, for example).
        my_exceptions : list[Exception]
            List that stores any exceptions encountered during the initialization processes.
        
        Returns
        -------
        bool
            True if this initialization method succeeded, False otherwise.
        ~~~

    `_initialize_path_info(self, base_directory_name: str, full_inlist_path: pathlib.Path | None=None, base_dir: str | None=None, inlist_name: str | None=None, inlist_dir: str | None=None, inlist_suffix: str='in') -> None`
    :   Initialization method used to store the relevant path information in this generic parser object.
        ~~~
        Parameters
        ----------
        base_directory_name : str
            Name of the base directory (of your github repo in which this parser is used, for example).
        full_inlist_path : Path | None, optional
            Full Path (object) to the inlist file or None; by default None.
        base_dir : str | None, optional
            The base directory in which the inlists directory can be found; by default None.
        inlist_name : str | None, optional
            The name of the inlist file; by default None.
        inlist_dir : str | None, optional
            The name of the inlists directory in the base directory; by default None. If None, and 
            initialization mode 1 is not used (i.e. 'full_inlist_path' is None), the parsing object
            looks for a directory named 'inlists/user_input/'; by default None.
        inlist_suffix : str, optional
            The suffix of the inlist file; by default 'in'.

        Raises
        ------
        ExceptionGroup
            Any of the exceptions raised during the initialization methods will be stored in
            this exception group, which will be raised if none of the two initialization methods succeed.
        ~~~

{% include button_back.html %}
