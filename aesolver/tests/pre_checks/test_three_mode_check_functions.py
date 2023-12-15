'''Pytest module used to test the pre-check functions in the aesolver.pre_checks module.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
'''
# import statements
import pytest
import typing
import numpy as np
# import aesolver.precheck module
# --> used to retrieve check functions
import aesolver.pre_checks as my_checks
# import pytest util classes
from pytest_util_classes import EVs, EPPs


# set up the input data for the linear driving rates
class LinDrivRates(EVs):
    """Contains the linear driving rates used for testing the check functions."""
    
    FIRST = np.array([1.0, 2.0, 3.0]).tobytes()
    SECOND = np.array([-1.0, 2.0, 3.0]).tobytes()
    THIRD = np.array([1.0, -2.0, -3.0]).tobytes()
    FOURTH = np.array([-1.0, -2.0, -3.0]).tobytes()


# set up the input parameters used to
# test the 'check_direct' function
class InputParamsDirect(EPPs):
    """Input parameters for the 'check_direct' function."""
    
    FIRST = "Daughter driving rate is positive."
    SECOND = None
    THIRD = "Daughter driving rate is positive; " \
            "parent driving rates are negative."
    FOURTH = "Parent driving rates are negative."


# set up the input parameters used to
# test the 'check_direct' function
class InputParamsDriven(EPPs):
    """Input parameters for the 'check_driven' function."""
    
    FIRST = None
    SECOND = "One driving rate is negative."
    THIRD = "Two driving rates are negative."
    FOURTH = "All driving rates are negative."
    

# set up the input parameters used to
# test the 'check_direct' function
class InputParamsParametric(EPPs):
    """Input parameters for the 'check_parametric' function."""
    
    FIRST = "Daughter driving rates are positive."
    SECOND = "Parent driving rate is negative; " \
             "Daughter driving rates are positive."
    THIRD = None
    FOURTH = "Parent driving rate is negative."


# set up fixture for the testing class
@pytest.fixture(
    scope='class',
    params=[(InputParamsDirect, "direct"),
            (InputParamsDriven, "driven"),
            (InputParamsParametric, "parametric")
            ]
    )
def setup_test_class_input(request) -> None:
    """Generates a fixture for the testing class."""
    # get the input value dictionary
    request.cls.iv_dict: dict[str, typing.Any] = \
        LinDrivRates.get_enumeration_dict()
    # get the input parameter dictionary
    request.cls.inp_dict: dict[str, typing.Any] = \
        {
            y: x
            for x, y in zip(
                request.param[0].get_all_pytest_parameters(),
                ['first', 'second', 'third', 'fourth']
                )
        }
    # get the function string
    request.cls.function_string: str = request.param[1]


# define the testing class for testing the functions
@pytest.mark.parametrize(
    'my_input', ['first', 'second', 'third', 'fourth']
    )
@pytest.mark.usefixtures('setup_test_class_input')
class TestThreeModeFunctionChecks:
    """Python test class for the three-mode-checks."""
    
    # attribute type declarations
    iv_dict: dict[str, typing.Any]
    inp_dict: dict[str, typing.Any]
    function_string: str
    
    def test_check_function(self, my_input: str) -> None:
        """Tests a check function with the parametrized input.
        
        Parameters
        ----------
        my_input : str
            The input string referring to one of the four input values.
        """
        # get the input parameter and value for the test
        input_value = {"linear driving rates": self.iv_dict[my_input]}
        input_parameter = self.inp_dict[my_input]
        # check if the parameter is marked as failing
        if len(input_parameter.marks) == 1:
            pytest.xfail(input_parameter.marks[0].kwargs['reason'])
        # assert if the check function obtains the correct output
        else:
            # use the 'getattr' function to retrieve the check function
            assert getattr(my_checks, f"check_{self.function_string}")(
                mode_info_dict=input_value
                )
