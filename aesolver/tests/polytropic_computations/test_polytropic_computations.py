'''Pytest file used to test the polytropic computations module of aesolver.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
'''
# import statements
import pytest
import typing
import numpy as np
# import support module
from expected_values import \
    ExpectedValuesPolytropicComputations as EvPC
# import mocking function/class
from mock_structure_data import \
    GYREPolytropeStructureValuesAfterLoad as GPSval
# import the class to be tested
from aesolver.polytropic_computations import PaSi


# comparison function for dictionaries for GYRE Profile
def compare_dicts(first: dict[str, typing.Any],
                  second: dict[str, typing.Any]) -> bool:
    """Compares the contents of two dictionaries.

    Parameters
    ----------
    first : dict[str, typing.Any]
        The first dictionary.
    second : dict[str, typing.Any]
        The second dictionary.
        
    Returns
    -------
    bool
        True if the contents of both dictionaries are comparable, False, if not.
    """
    # check if lengths are the same
    check_len = len(first) == len(second)
    # initialize the comparable bool to True
    # (if multiplied with False, it will become False)
    if check_len:
        comparable = True
        # loop through the first dictionary and compare values with
        # those of the second dictionary
        for _k, _v in first.items():
            if isinstance(_v, np.ndarray):
                    comparable *= np.allclose(
                        _v, second[_k]
                        )
            else:
                comparable *= _v == second[_k]
        # if both comparable and lengths are same, return True
        return comparable
    else:
        return False


# fixture used to set up test class
@pytest.fixture(scope='class')
def setup_test_class(request) -> None:
    """Setup fixture for the test class."""
    # store the mock data
    request.cls.mock_data: list[dict[str, typing.Any]] \
        = [GPSval.get_dict()]
    # store the model mass and model radius
    request.cls.model_mass: float = 3.0
    request.cls.model_radius: float = 5.0
    # store the expected values dictionary
    request.cls.expect_vals: dict[str, typing.Any] \
        = EvPC.get_enumeration_dict()

# define the test class
@pytest.mark.usefixtures('setup_test_class')
class TestPolytropicComputations:
    """Tests the polytropic computations module of aesolver."""
    
    # attribute type declarations
    mock_data: list[dict[str, typing.Any]]
    model_mass: float
    model_radius: float
    expect_vals: dict[str, typing.Any]
    
    def test_poly_initialize_comp_obj(self) -> None:
        """Tests initialization of the polytropic computation object."""
        # initialize the object
        my_obj = PaSi(
            model_info_dictionary_list=self.mock_data,
            model_mass=self.model_mass,
            model_radius=self.model_radius
            )
        # assert it is initialized
        assert my_obj
    
    def test_poly_comps(self) -> None:
        """Tests the polytropic computations."""  
        # initialize the object
        my_obj = PaSi(
            model_info_dictionary_list=self.mock_data,
            model_mass=self.model_mass,
            model_radius=self.model_radius
            )
        # compare input and output
        assert compare_dicts(
            first=my_obj.property_dict,
            second=self.expect_vals
            )
