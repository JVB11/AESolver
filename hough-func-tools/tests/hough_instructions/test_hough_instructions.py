"""Pytest module used to test the Hough-instructions module.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import modules
import pytest
import json

# import module to be tested
from hough_function.hough_instructions import InstructionsEnum

# import pytest utility classes
from pytest_util_classes import EVs, EPPs  # type: ignore


# set up the input variables
class InputInstructions(EVs):
    """Contains the input instruction fields."""

    FIRST = 'PRAT'
    SECOND = 'LEE_SAIO_EQUIVALENT'
    THIRD = 'TOWNSEND'
    FOURTH = 'UNKNOWN'


# define whether these instructions fail or not
class FailedInstructions(EPPs):
    """Determines whether instructions fail."""

    FIRST = None
    SECOND = None
    THIRD = None
    FOURTH = 'Unknown instruction!'


# define the expected output for the instructions
class OutputInstructions(EVs):
    """Contains the output instructions."""

    FIRST = json.dumps(
        {'adjust_phi_sign': False, 'sin_factor': False}, sort_keys=True
    )
    SECOND = json.dumps(
        {'adjust_phi_sign': True, 'sin_factor': False}, sort_keys=True
    )
    THIRD = json.dumps(
        {'adjust_phi_sign': True, 'sin_factor': True}, sort_keys=True
    )
    FOURTH = None


# set up a fixture for the test class
@pytest.fixture(
    scope='class', params=FailedInstructions.get_all_pytest_parameters()
)
def setup_test_class(request):
    """Sets up the fixture for the test class."""
    request.cls.input_val = InputInstructions.get_value(request.param)
    request.cls.output_val = OutputInstructions.get_value(request.param)


# set up a test class
@pytest.mark.usefixtures('setup_test_class')
class TestHoughInstructions:
    """Tests the Hough-instructions module."""

    # attribute type declarations
    input_val: str
    output_val: dict

    def test_get_instructions(self):
        """Test the get_instructions method."""
        # retrieve the instructions based on the input
        my_instructions = InstructionsEnum.get_instructions(self.input_val)
        # assert these instructions are correct
        assert my_instructions == self.output_val
