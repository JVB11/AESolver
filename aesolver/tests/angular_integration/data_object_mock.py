'''Defines a data object class to be used for testing the HoughIntegrate module of aesolver.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
'''
# import statements
import numpy as np
from enum import Enum


# define the mock-up data object class
class MockDataObj:
    """Mock-up data object class to be used for testing purposes."""
    
    # attribute type declarations
    _nr_datasets: int
    _mu_values: np.ndarray
    my_data_arr1: np.ndarray
    my_data_arr2: np.ndarray
    my_data_arr3: np.ndarray
    _theta_values: np.ndarray
    
    def __init__(self, my_input_enum: Enum,
                 max_nr_datasets: int=3) -> None:
        # initialize the parameter holding how many datasets
        # are to be used (MAX 3)
        self._nr_datasets = max_nr_datasets
        # initialize the input dat based on an enum
        self._unpack_enum_values(my_enum=my_input_enum)
        # make sure you have owned types of arrays!
        self._owned()

    def _owned(self) -> None:
        """Ensures all enum-read bytes/arrays are writable by making owned copies!"""
        # NECESSARY FOR C++ integration actions!
        self._mu_values = np.array(self._mu_values)
        self.my_data_arr1 = np.array(self.my_data_arr1)
        self.my_data_arr2 = np.array(self.my_data_arr2)
        self.my_data_arr3 = np.array(self.my_data_arr3)

    def _unpack_enum_values(self, my_enum: Enum) -> None:
        """Unpack the stored enumerated input values and keys and store them in this data object.

        Parameters
        ----------
        my_enum : Enum
            The enumeration object holding the input values.
        """
        for _k, _v in my_enum.access_all_input_data(
            max_nr_data=self._nr_datasets
            ).items():
            setattr(self, _k, _v)
