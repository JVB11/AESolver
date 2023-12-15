'''Pytest module (unit-)testing the three-mode stability check functions delivered by aesolver.stability_checker.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
'''
# import statements
import pytest
import json
import numpy as np
# import pytest util class
from pytest_util_classes import EVs
# import functionalities to be tested
from aesolver.stability_checker import \
    gss, check_van_beeck_conditions


# GSS CONDITIONS

    
# store the input values for the test
class InputGSS(EVs):
    """Input values used to test gss functionality."""
    
    NR_1 = np.array([1.e-7, -2.e-7, -2.5e-7]).tobytes()
    NR_2 = json.dumps([1.e-7, -2.e-7, -2.5e-7])
    NR_3 = -3.5e-7
    NR_4 = np.array([-1.e-7, 2.e-7, 2.5e-7]).tobytes()
    NR_5 = json.dumps([-1.e-7, 2.e-7, 2.5e-7])
    NR_6 = 3.5e-7    


# store the output values for the test
class OutputGSS(EVs):
    """Output values used to test gss functionality."""
    
    NR_1 = True
    NR_2 = True
    NR_3 = True
    NR_4 = False
    NR_5 = False
    NR_6 = False


# set up a fixture for the test class
@pytest.fixture(scope="class", params=range(1,7))
def setup_test_class_gss(request) -> None:
    # store the input value
    request.cls.input_val: np.ndarray | float | list \
        = InputGSS.get_value(f"nr_{request.param}")
    # store the expected output value
    request.cls.expected: bool = OutputGSS.get_value(
        f"nr_{request.param}"
        )


# define the test class
@pytest.mark.usefixtures('setup_test_class_gss')
class TestGSS:
    """Tests the gss functionality."""
    
    # attribute type declarations
    input_val: np.ndarray | float | list
    expected: bool
    
    def test_gss(self) -> None:
        # obtain the result
        my_result = gss(self.input_val)
        # assert the result is OK
        assert my_result == self.expected


# VAN BEECK CONDITIONS

    
# store the input values for the test
class InputVB(EVs):
    """Input values used to test gss functionality."""
    
    NR_1_GAMMAS = np.array(
        [1.e-7, -2.e-7, -3.0e-7]
        ).tobytes()  # rho2 = 2, rho3 = 3, rho23 = 6, rhos = -4
        # coeffq0 = -76, coeffq2 = 40, coeffq4 = 36
    NR_2_GAMMAS = np.array(
        [1.e-7, -2.e-7, -3.0e-7]
        ).tobytes()  # rho2 = 2, rho3 = 3, rho23 = 6, rhos = -4
        # coeffq0 = -76, coeffq2 = 40, coeffq4 = 36
    NR_3_GAMMAS = np.array(
        [1.e-7, -4.e-7, -5.0e-7]
        ).tobytes()  # rho2 = 4, rho3 = 5, rho23 = 20, rhos = -8
        # coeffq0 = -552, coeffq2 = 432, coeffq4 = 120
    NR_4_GAMMAS = np.array(
        [2.e-7, -2.e-7, -4.0e-7]
        ).tobytes()  # rho2 = 1, rho3 = 2, rho23 = 2, rhos = -2
        # coeffq0 = -12, coeffq2 = 0 , coeffq4 = 12
    NR_1_Q = 5.0  # q**2 = 25, q**4 = 625
    NR_2_Q = 10.0  # q**2 = 100, q**4 = 10000
    NR_3_Q = 5.0  # q**2 = 25, q**4 = 625
    NR_4_Q = 1.0  # q**2 = 1, q**4 = 1


# store the output values for the test
class OutputVB(EVs):
    """Output values used to test gss functionality."""
    
    NR_1 = True
    NR_2 = True
    NR_3 = True
    NR_4 = False


# set up a fixture for the test class
@pytest.fixture(scope="class", params=range(1,5))
def setup_test_class_vb(request) -> None:
    # store the input values
    request.cls.gammas: np.ndarray = InputVB.get_value(
        f"nr_{request.param}_gammas"
        )
    request.cls.q: float = InputVB.get_value(
        f'nr_{request.param}_q'
        )
    # store the expected output value
    request.cls.expected: bool = OutputVB.get_value(
        f"nr_{request.param}"
        )    


# define the test class
@pytest.mark.usefixtures('setup_test_class_vb')
class TestVBConditions:
    """Tests the 'check_van_beeck_conditions' functionality."""
    
    # attribute type declarations
    gammas: np.ndarray
    q: float
    expected: bool
    
    def test_check_van_beeck_conditions(self) -> None:
        # obtain the result
        my_result = check_van_beeck_conditions(
            gammas=self.gammas, q=self.q
            )
        # assert the result is OK
        assert my_result == self.expected
