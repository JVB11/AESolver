'''Python test module for the input generator class for Adiabatic-Nonadiabatic mode comparison.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
'''
# import statements
import pytest
import typing
import numpy as np
# import from modules
from aesolver.mode_input_generator.\
    input_generator import AdNadModeInfoInputGenerator
from pytest_util_classes import EVs, EPPs, EFPVs, IIFTs


# generate enumeration object containing the pytest parameters for the tests
class InputGeneratorPytestParameters(EPPs):
    """Enumeration object containing the pytest parameters for the tests."""
    
    # enumeration pytest parameters
    RIGHT = None
    WRONG_M = 'Wrong output m value.'
    WRONG_L = 'Wrong output l value.'
    WRONG_COMBOS = 'Wrong combinations.'
    WRONG_AVG_ORDERS = 'Wrong average radial orders.'
    WRONG_DIFFERENCE_STATISTICS = \
        'Wrong Wu & Goldreich (2001) difference statistics.'


# generate enumeration object containing the changed parameters for the tests
class InputGeneratorFailingParameterValues(EFPVs):
    """Enumeration object containing the failing parameter values for the tests."""
    
    RIGHT = (None, None)
    WRONG_M = ('mode_m', 3)
    WRONG_L = ('mode_l', 3)
    WRONG_COMBOS = \
        ('combos', [[20, 15, 16], [22, 15, 16]])
    WRONG_AVG_ORDERS = \
        ('avg_radial_orders', [17, 17.5])
    WRONG_DIFFERENCE_STATISTICS = \
        ('wu_difference_statistics', [19, 25])


# generate enumeration object for default test input values
class CommonInputValues(EVs):
    """Contains the common/default test input values."""
    
    DAMPED_N_LOW1 = 15
    DAMPED_N_LOW2 = 15
    DAMPED_N_HIGH1 = 16
    DAMPED_N_HIGH2 = 16
    DRIVEN_N_LOW = 20
    DRIVEN_N_HIGH = 21
    MODE_M = 2
    MODE_L = 2


# generate enumeration object for the expected output values
class ExpectedOutputValues(EVs):
    """Contains the expected test output values."""
    
    MODE_L = 2
    MODE_M = 2
    COMBOS = [[20, 15, 16], [20, 16, 15], [20, 16, 16], [20, 15, 15],
              [21, 15, 16], [21, 16, 15], [21, 16, 16], [21, 15, 15]]
    AVG_RADIAL_ORDERS = [17., 17., 17.33333333, 16.66666667, 17.33333333,
                         17.33333333, 17.66666667, 17.]
    WU_DIFFERENCE_STATISTICS = [19, 19, 20, 20, 20, 20, 21, 21]


# generate the fixture that will be used for the class setup
@pytest.fixture(scope="class",
                params=InputGeneratorPytestParameters.get_all_pytest_parameters())
def setup_test_class(request) -> None:
    """Generates a fixture for the testing class so that information is available for testing."""
    # get the setup object
    _my_setup_object: IIFTs = IIFTs(
        common_input_dict=CommonInputValues.get_enumeration_dict(),
        expected_output_dict=ExpectedOutputValues.get_enumeration_dict()
        )
    # get the correct input and output dictionary
    _my_output_dict: dict[str, typing.Any] = \
        _my_setup_object.get_output_dictionary(
            *InputGeneratorFailingParameterValues.\
                get_failing_parameter_and_value(
                    request.param
                    )
            )
    # set the input class variables
    request.cls.generator_object: AdNadModeInfoInputGenerator = \
        AdNadModeInfoInputGenerator(
            **_my_setup_object.input_dictionary
            )
    # set the output class variables
    request.cls.mode_l_output: int = \
        _my_output_dict['mode_l']
    request.cls.mode_m_output: int = \
        _my_output_dict['mode_m']
    request.cls.combos_output: list | np.ndarray = \
        _my_output_dict['combos']
    request.cls.avg_rad_orders_output: list | np.ndarray = \
        _my_output_dict['avg_radial_orders']
    request.cls.wu_difference_statistics_output: \
        list | np.ndarray = \
            _my_output_dict['wu_difference_statistics']


# generate the test class
@pytest.mark.usefixtures("setup_test_class")
class TestInputGenerator:
    """Python test class for the input generator."""
    
    # attribute type declarations
    generator_object: AdNadModeInfoInputGenerator
    mode_l_output: int
    mode_m_output: int
    combos_output: list | np.ndarray
    avg_rad_orders_output: list | np.ndarray
    wu_difference_statistics_output: list | np.ndarray

    def test_correct_mode_geometry(self) -> None:
        """Tests for correct mode geometry."""
        assert all([self.generator_object.mode_l == self.mode_l_output,
                    self.generator_object.mode_m == self.mode_m_output])
    
    def test_combinations(self) -> None:
        """Tests for correct input combinations."""
        assert all(
            [_combo == _output_combo for _combo, _output_combo
             in zip(self.generator_object.combinations_radial_orders,
                    self.combos_output)]
            )

    def test_average_radial_orders(self) -> None:
        """Tests for correct average radial orders."""
        assert np.allclose(self.generator_object.average_radial_orders,
                           self.avg_rad_orders_output)
        
    def test_wu_difference_statistics(self) -> None:
        """Tests for correct Wu difference statistics."""
        assert np.equal(self.generator_object.wu_difference_statistics,
                        self.wu_difference_statistics_output).all()
