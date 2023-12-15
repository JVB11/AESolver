'''Pytest module that tests the enumeration class 'EnumRadial' of the aesolver.coupling_coefficients module

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
'''
# import statements
import pytest
# import support modules

# import expected value modules

# import module to be tested
from aesolver.coupling_coefficients.\
    coefficient_support_classes import CCR


# fixture for the test class
@pytest.fixture(scope='class')
def setup_test_class(request) -> None:
    pass

# define the test class
@pytest.mark.usefixtures('setup_test_class')
class TestEnumRadial:
    """_summary_"""
    
    # attribute type declarations
