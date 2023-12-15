'''Pytest module that tests the enumeration class 'ModeData' of the aesolver.coupling_coefficients module

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
'''
# import statements
import pytest
# import support modules

# import expected value modules
from expected_values_enumerated_mode_data \
    import expected_vals_mode_independent_names
# import module to be tested
from aesolver.coupling_coefficients.\
    enumeration_files import ModeData


# fixture for the test class
@pytest.fixture(scope='class')
def setup_test_class(request) -> None:
    # store the expected output
    # of the 'get_common_attrs' method.
    request.cls.output_common_attrs: list[str] = \
        expected_vals_mode_independent_names()

# define the test class
@pytest.mark.usefixtures('setup_test_class')
class TestModeData:
    """Tests the ModeData enumeration module."""
    
    # attribute type declarations
    output_common_attrs: list[str]
    
    def test_get_common_attrs(self):
        """Tests the 'get_common_attrs' method of ModeData."""
        # get the output from the method
        my_out_list = ModeData.get_common_attrs()
        # assert it is ok
        assert my_out_list == \
            self.output_common_attrs
    