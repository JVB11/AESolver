Module pytest_util_classes
==========================
Initialization file for the python package containing utility classes that facilitate the writing of pytest tests.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>

Sub-modules
-----------
* pytest_util_classes.data_classes
* pytest_util_classes.enumeration_classes

Classes
-------

`EFPVs(new_class_name, names, *, module=None, qualname=None, type=None, start=1, boundary=None)`
:   Enumeration object containing the failing parameter values for the tests.

    ### Ancestors (in MRO)

    * enum.Enum

`EPPs(new_class_name, names, *, module=None, qualname=None, type=None, start=1, boundary=None)`
:   Enumeration object containing the pytest parameters for the tests.

    ### Ancestors (in MRO)

    * enum.Enum

`EVs(new_class_name, names, *, module=None, qualname=None, type=None, start=1, boundary=None)`
:   Enumeration class used to store common input
    and expected output values for tests.

    ### Ancestors (in MRO)

    * enum.Enum

`IIFTs(common_input_dict: dict, expected_output_dict: dict)`
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