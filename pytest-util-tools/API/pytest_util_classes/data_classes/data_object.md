Module pytest_util_classes.data_classes.data_object
===================================================
Python module containing superclass for data objects used for testing various modules.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>

Classes
-------

`InputInformationForTests(common_input_dict: dict, expected_output_dict: dict)`
:   Information object/class used to access relevant information for the tests.

    ### Instance variables

    `expected_output_dictionary`
    :   Returns the dictionary containing the expected output arguments.

    `input_dictionary`
    :   Returns the input dictionary.
        
        Returns
        -------
        dict
            Dictionary containing the (common) input values.

    ### Methods

    `get_output_dictionary(self, mykey: None, myval: None) -> dict`
    :   Constructs and returns the output dictionary for a specific test case.
        
        Parameters
        ----------
        mykey : None
            No valid key, so default values are returned.
        myval : None
            Dummy argument, because no values are changed.
        
        Returns
        -------
        dict
            The output dictionary containing the default arguments.