'''Pytest module (unit-)testing the two-mode stability check functions delivered by aesolver.stability_checker.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
'''
# import statements
import pytest
import numpy as np
# import pytest util class
from pytest_util_classes import EVs
# import functionalities to be tested
from aesolver.stability_checker import \
    check_dziembowski_direct, \
        check_dziembowski_parametric


# store the input values for the test
class InputValues(EVs):
    """Stores the input values for the test."""
    
    FIRST_GAMMAS = np.array(
        [1.e-7, -2.e-7, -2.e-7]  # theta = 2
        ).tobytes()
    FIRST_Q = 5.0
    SECOND_GAMMAS = np.array(
        [1.e-7, -3.0e-7, -3.0e-7]  # theta = 3
        ).tobytes()
    SECOND_Q = 5.0
    THIRD_GAMMAS = np.array(
        [1.e-7, -2.e-7, -2.e-7]  # theta = 2
        ).tobytes()
    THIRD_Q = 1.0
    FOURTH_GAMMAS = np.array(
        [-1.e-7, 0.3e-7, 0.3e-7]  # theta = 0.3
        ).tobytes()
    FOURTH_Q = 5.0
   
 
# store the output booleans for the test
class OutputValues(EVs):
    """Stores the result values for the test."""
    
    FIRST_DIRECT = False
    FIRST_PARAMETRIC = True
    SECOND_DIRECT = False
    SECOND_PARAMETRIC = True
    THIRD_DIRECT = False
    THIRD_PARAMETRIC = True
    FOURTH_DIRECT = True
    FOURTH_PARAMETRIC = False


# set up a fixture for the test class
@pytest.fixture(
    scope="class",
    params=["first", "second", "third", "fourth"]
    )
def setup_test_class(request) -> None:
    # get the enumeration dictionary for the output values
    _out_dict = OutputValues.get_enumeration_dict()
    # store the expect output boolean values
    request.cls.expect_direct: bool = _out_dict[
        f'{request.param}_direct'
        ]
    request.cls.expect_parametric: bool = _out_dict[
        f'{request.param}_parametric'
        ]
    # get the input values
    request.cls.gammas: np.ndarray = InputValues.get_value(
        f'{request.param}_gammas', dtype_buffered=np.float64
        )
    request.cls.q: float = InputValues.get_value(
        f'{request.param}_q'
        )   


# define the test class
@pytest.mark.usefixtures('setup_test_class')
class TestStabilityCheckFunctionsTwoMode:
    """Test class for the stability check functions in the two mode case."""
    
    # attribute type declarations
    expect_direct: bool
    expect_parametric: bool
    gammas: np.ndarray
    q: float
    
    def test_direct(self) -> None:
        """Tests the direct stability check function."""
        # get the stability check result
        stab_check = check_dziembowski_direct(
            gammas=self.gammas
            )
        # assert the result is correct
        assert self.expect_direct == stab_check
        
    def test_parametric(self) -> None:
        """Tests the parametric stability check function."""
        # get the stability check result
        stab_check = check_dziembowski_parametric(
            gammas=self.gammas, q=self.q
            )
        # assert the result is correct
        assert self.expect_parametric == stab_check
    