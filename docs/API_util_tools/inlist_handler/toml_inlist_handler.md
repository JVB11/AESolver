Module inlist_handler.toml_inlist_handler
=========================================
Python module that defines the class needed to parse toml-style inlists and retrieves the necessary information.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>

Classes
-------

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