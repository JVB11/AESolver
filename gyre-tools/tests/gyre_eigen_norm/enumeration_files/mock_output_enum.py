"""Contains code to generate mock output data for our test of this package.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import logging

# import custom pytest utility class
from pytest_util_classes import EVs  # type: ignore


# set up logger
logger = logging.getLogger(__name__)


# create output data subclass
class OutputData(EVs):
    """Contains the mock-up output data."""

    # enumeration member definitions
    ONES = str({'lee2012': 0.398863, 'schenk2002': 0.564077})

    # retrieve the output data for a specific InputData class
    @classmethod
    def get_output(cls, my_input_class, my_norm_method):
        """Retrieves the output data associated with a given InputData class.

        Parameters
        ----------
        my_input_class : InputDataEnum
            The class of input data.
        my_norm_method : str
            The normalization method used.

        Returns
        -------
        my_val : list or np.ndarray or float
            The expected output data associated with a given InputData class/set.
        """
        # retrieve the string representation of the input class
        _input_repr = my_input_class.get_custom_repr()
        # use that string representation to get the associated output data
        return eval(cls.get_value(_input_repr.upper()))[my_norm_method.lower()]
