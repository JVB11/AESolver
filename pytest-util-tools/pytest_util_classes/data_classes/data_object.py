"""Python module containing superclass for data objects used for testing various modules.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import numpy as np
from copy import copy
from multimethod import multimethod  # type: ignore


# input information class
class InputInformationForTests:
    """Information object/class used to access relevant information for the tests."""

    # attribute type declarations
    common_input_dict: dict
    expected_output_dict: dict

    def __init__(
        self, common_input_dict: dict, expected_output_dict: dict
    ) -> None:
        # store the common input dictionary and expected output dictionary
        self.common_input_dict = common_input_dict
        self.expected_output_dict = expected_output_dict

    @property
    def input_dictionary(self) -> dict:
        """Returns the input dictionary.

        Returns
        -------
        dict
            Dictionary containing the (common) input values.
        """
        return self.common_input_dict

    @property
    def expected_output_dictionary(self) -> dict:
        """Returns the dictionary containing the expected output arguments."""
        return self.expected_output_dict

    @multimethod
    def get_output_dictionary(self, mykey: None, myval: None) -> dict:
        """Constructs and returns the output dictionary for a specific test case.

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
        """
        return self.expected_output_dictionary

    @get_output_dictionary.register(str, int)  # type: ignore
    @get_output_dictionary.register(str, float)  # type: ignore
    @get_output_dictionary.register(str, list)  # type: ignore
    @get_output_dictionary.register(str, np.ndarray)  # type: ignore
    @get_output_dictionary.register(str, str)  # type: ignore
    def _(
        self, mykey: str, myval: int | float | list | np.ndarray | list
    ) -> dict:
        """The registered multimethod that replaces the necessary value.

        Parameters
        ----------
        mykey : str
            The key for the value that needs to be replaced.
        myval : int or float or list or np.ndarray or str
            The replacement value.

        Returns
        -------
        replacement_dict : dict
            The dictionary with the value replaced.
        """
        replacement_dict = copy(self.expected_output_dictionary)
        replacement_dict[mykey] = myval
        return replacement_dict
