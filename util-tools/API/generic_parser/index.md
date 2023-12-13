Module generic_parser
=====================
Python package initialization file for package containing a generic class that can be subclassed to provide a parser used to parse arguments from the command line and/or an inlist.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>

Sub-modules
-----------
* generic_parser.generic_parser

Classes
-------

`GenericParser(base_directory_name: str, full_inlist_path: pathlib.Path | None = None, base_dir: str | None = None, inlist_name: str | None = None, inlist_dir: str | None = None, inlist_suffix: str = 'in')`
:   (Super)Class containing generic methods and arguments used for parsing of input.
    Input can either be read from the command-line argument parser, or from inlist files.
    
    Notes
    -----
    This class can be initialized in two different ways:
    
    1. Either you provide the full (pathlib) Path to the file that you want to parse from, or,
    2. you provide the strings representing the base directory, inlist name, inlist directory, and inlist suffix.
    
    If you do not provide the necessary information for one of these initialization methods, the initialization will fail with an ExceptionGroup that denotes the failures in understanding the user input.
    
    Parameters
    ----------
    base_directory_name : str
        Name of the base directory (of your github repo in which this parser is used, for example).
    full_inlist_path : Path | None, optional
        Initialization mode 1: full (pathlib) Path to the file you want to parse from, by default None (i.e. opting to perform actions in this class using initialization option 2).
    base_dir : str | None, optional
        Initialization mode 2: The base directory in which the inlists directory can be found, by default None (i.e. opting to perform actions in this class using initialization option 1).
    inlist_name : str | None, optional
        Initialization mode 2: The name of the inlist file, by default None (i.e. opting to perform actions in this class using initialization option 1).
    inlist_dir : str or None, optional
        Initialization mode 2: The name of the inlists directory in the base directory.
        If None, and initialization mode 1 is not used (i.e. 'full_inlist_path' is None), the parsing object looks for a directory named 'inlists/user_input/'; by default None.
    inlist_suffix : str, optional
        Initialization mode 2: The suffix of the inlist file (only used for initialization mode 2), by default 'in'.

    ### Instance variables

    `inlist_path`
    :   Returns the path to the used/selected inlist.
        
        Returns
        -------
        str
            Path to the used/selected inlist.

    ### Methods

    `get_ha(self, ha_dict, index_val)`
    :   Hierarchical indexing utility method used to access
        (possibly nested) dictionary elements.
        
        Parameters
        ----------
        ha_dict : dict
            (Possibly nested) dictionary containing input elements.
        index_val : list[str]
            Index list for the (possibly nested) dict.
        
        Returns
        -------
        int or str or float or list
            Requested value.

    `read_args(self)`
    :   Generic method used to read relevant arguments from the parsed information.

    `read_custom_args(self)`
    :   Method used to read relevant arguments from the information parsed from
        a custom inlist.
        
        Notes
        -----
        Meant to be overloaded by its subclass.

    `read_in_to_dict(self, tuple_list)`
    :   Generic method used to read list of tuples of arguments in to a
        dictionary.
        
        Notes
        -----
        If a(n optional) third value is supplied in the tuple list,
        it should be provided for all input arguments.
        Use False for the negating bool variable (optional third entry)
        for non-boolean input!
        
        Parameters
        ----------
        tuple_list : list[tuple]
            List containing tuples that represent elements of the dictionary:
            First element is the dictionary key. The second element is
            the dictionary value. The optional third element is a boolean
            that inverses a boolean dictionary value if True.
        
        Returns
        -------
        my_data_dict : dict
            The dictionary containing the loaded items.

    `read_in_to_list(self, argument_list)`
    :   Generic method used to read list of arguments in to a list.
        
        Notes
        -----
        If a(n optional) second value is supplied in the argument (tuple) list,
        it should be provided for all input arguments.
        Use False for the negating bool variable (optional second entry)
        for non-boolean input!
        
        Parameters
        ----------
        argument_list : list[str] or list[tuple]
            List containing arguments that represent the names of the arguments
            read in the inlist or argumentparser. Optional second (tuple) value
            is a boolean that inverses a boolean value if True.
        
        Returns
        -------
        list
            The list of data containing the requested parsed arguments values.

    `read_toml_args(self)`
    :   Method used to read relevant arguments from the parsed
        toml inlist file.
        
        Notes
        -----
        Meant to be overloaded by its subclass.