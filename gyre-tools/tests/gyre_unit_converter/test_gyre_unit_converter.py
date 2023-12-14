"""Python pytest module for the GYREFreqUnitConverter enumeration class.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import pytest
import random

# import the GYRE unit converter functionality
from gyre_unit_converter import GYREFreqUnitConverter

# import utility classes for pytest
from pytest_util_classes import EFPVs, EPPs, EVs  # type: ignore


# generate enumeration object containing the pytest parameters for the tests
class UnitConverterPytestParameters(EPPs):
    """Enumeration object containing the pytest parameters for the tests."""

    # enumeration pytest parameters
    RIGHT = None
    WRONG_CONVERSION = None


# generate enumeration object containing the changed parameters for the tests
class UnitConverterFailingParameterValues(EFPVs):
    """Enumeration object containing the failing parameter values for the tests."""

    RIGHT = (None, None)
    WRONG_CONVERSION = ('cyc', 'cycles_per_years')


# generate enumeration object for default test input values
class CommonInputValues(EVs):
    """Contains the common/default test input values."""

    TEXT = random.choice(
        [
            _conversion_key
            for _conversion_key in GYREFreqUnitConverter.__members__.keys()
        ]
    )


# generate enumeration object for expected output values
class ExpectedOutputValues(EVs):
    """Contains the expected test output values."""

    CYC_PER_DAY = r'd$^{-1}$'
    HZ = r'Hz'
    UHZ = r'$\mu$Hz'
    RAD_PER_SEC = r'rad s$^{-1}$'


# generate the fixture that will be used for the class setup
@pytest.fixture(
    scope='class',
    params=UnitConverterPytestParameters.get_all_pytest_parameters(),
)
def setup_test_class(request):
    """Generates a fixture for the testing class so that information is available for testing."""
    # get the input for the test
    _input_val = CommonInputValues.get_value(my_key='text')
    _expected_output_val = ExpectedOutputValues.get_value(my_key=_input_val)
    _output_val = (
        _expected_output_val
        if request.param == 'RIGHT'
        else UnitConverterFailingParameterValues.get_failing_parameter_and_value(
            name=request.param
        )[1]
    )
    # set the input class variable
    request.cls.conversion_value = GYREFreqUnitConverter.convert(
        text=_input_val
    )
    # set the output class variable
    request.cls.output_value = _output_val
    # store the node to add specific reason
    request.cls.node = request.node


# generate the test class
@pytest.mark.usefixtures('setup_test_class')
class TestUnitConverter:
    """Python test class for the unit converter."""

    # attribute type declarations
    conversion_value: str
    output_value: str
    param: str

    def test_correct_conversion(self):
        """Test for correct unit conversion."""
        if not (comparison_bool := self.compare_values()):
            self.print_comparison_fail_message()
        assert comparison_bool

    def compare_values(self):
        """Compares the values of the conversion and (expected) output parameter.

        Returns
        -------
        bool
            True if they are the same, False if not.
        """
        return self.conversion_value == self.output_value

    def print_comparison_fail_message(self):
        """Prints comparison fail message."""
        self.node.add_marker(  # type: ignore
            pytest.mark.xfail(
                reason=f'Wrong conversion: {self.conversion_value} != {self.output_value}'
            )
        )
