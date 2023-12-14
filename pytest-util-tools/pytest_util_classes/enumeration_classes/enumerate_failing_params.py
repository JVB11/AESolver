"""Enumeration superclass module for failing parameters.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import sys
from enum import Enum


# generate enumeration object containing the changed parameters for the tests
class EnumerateFailingParameterValues(Enum):
    """Enumeration object containing the failing parameter values for the tests."""

    # retrieve the changed value corresponding to the input value
    @classmethod
    def get_failing_parameter_and_value(cls, name: str) -> tuple:
        """Get the failing parameter value, based on the pytest parameter name.

        Parameters
        ----------
        name : str
            The name of the pytest parameter for which a failing parameter needs to be returned.

        Returns
        -------
        tuple
            tuple containing the key and the value that need to be changed.
        """
        try:
            return cls.__getitem__(name).value
        except AttributeError:
            print(
                f'Attribute {name} does not exist in this enumeration object! Now exiting.'
            )
            sys.exit()
