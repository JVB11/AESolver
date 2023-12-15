'''Python module containing one enumeration class that holds the expected output values for our test.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
'''
# import pytest_util_class
from pytest_util_classes import EVs


# define the input parameters
class ExpectedOutputParameters(EVs):
    """Defines the expected output parameters for our tests."""
    
    TWO_DATA = 1.40325
    TWO_DATA_DIV_SIN = 2.31936
    TWO_DATA_MU = 0.691151
    TWO_DATA_PARALLEL = 1.40325
    TWO_DATA_CONJ = 1.40325
    TWO_DATA_CHECK_M = 1.40325
    TWO_DATA_LEN = 0.0
    THREE_DATA = 0.345575
    THREE_DATA_DIV_SIN = 0.814242
    THREE_DATA_MU = 0.253422
    THREE_DATA_PARALLEL = 0.345575
    THREE_DATA_CONJ = 0.345575
    THREE_DATA_CHECK_M = 0.345575
    THREE_DATA_LEN = 0.0
