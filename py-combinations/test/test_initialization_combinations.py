"""Pytest file used to test the initialization of the classes in the combinations module.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import modules
import pytest
import json
import numpy as np

# import pytest utility class
from value_subclass import EVsList

# from pytest_util_classes import EVs
# import custom modules to be tested
from py_combinations.cartesian_combinations import CartesianCombos
from py_combinations.single_set_combinations import SingleSetCombos
from py_combinations.symmetric_product import SymmetricProductCombos


# define the test inputs
class MyInputs(EVsList):
    """Input values for the tests."""

    TEST_LIST = json.dumps([15, 20, 30])
    TEST_LIST2 = json.dumps([24, 30])
    TEST_LIST3 = json.dumps([8])
    TEST_LIST_EMPTY = json.dumps([])
    TEST_LIST_FLOAT = json.dumps([2.5, 3.6])
    TEST_ARRAY = np.array([15.0, 20.0, 30.0]).tobytes()
    TEST_LIST_OK = json.dumps([[24, 25, 30], [15, 16]])
    TEST_INT = json.dumps(15)
    TEST_FLOAT = json.dumps(15.0)


# fixture used to provide inputs
@pytest.fixture(scope='class')
def init_params(request):
    # set the correct fixture class variable
    request.cls.input_list = MyInputs.get_list()


# test class for initialization of objects
@pytest.mark.usefixtures('init_params')
class TestInitialization:
    """Class containing test methods for initialization of the different classes."""

    @pytest.mark.parametrize(
        ('input_term_index', 'expected_fail'),
        [
            (i, x)
            for i, x in enumerate(
                [True, True, True, True, True, True, False, True, True]
            )
        ],
    )
    def test_symmetric_product(self, input_term_index, expected_fail):
        """Tests the initialization of the Symmetric product combos object.

        Parameters
        ----------
        input_term_index : int
            Index to retrieve the correct index for the test input fixture for the initialization of the 'SymmetricProductCombos' object.
        expected_fail : bool
            True if the fail is expected, False if not.
        """
        # retrieve the correct fixture value
        my_fixture_input = self.input_list[input_term_index]  # type: ignore
        # initialize the object (requires iterable input)
        if expected_fail:
            with pytest.raises(Exception) as e_info:
                print(
                    f'My initialization input: {my_fixture_input} (Expect exception being raised and captured!)'
                )
                _ = SymmetricProductCombos(*my_fixture_input)
        else:
            print(
                f'My initialization input: {my_fixture_input} (Expect test pass!)'
            )
            _ = SymmetricProductCombos(*my_fixture_input)

    @pytest.mark.parametrize(
        ('input_term_index', 'expected_fail'),
        [
            (i, x)
            for i, x in enumerate(
                [True, True, True, True, True, True, False, True, True]
            )
        ],
    )
    def test_cartesian(self, input_term_index, expected_fail):
        """Tests the initialization of the Cartesian combos object.

        Parameters
        ----------
        input_term_index : int
            Index to retrieve the correct index for the test input fixture for the initialization of the 'CartesianCombos' object.
        expected_fail : bool
            True if the fail is expected, False if not.
        """
        # retrieve the correct fixture value
        my_fixture_input = self.input_list[input_term_index]  # type: ignore
        # initialize the object (requires iterable input)
        if expected_fail:
            with pytest.raises(Exception) as e_info:
                print(
                    f'My initialization input: {my_fixture_input} (Expect exception being raised and captured!)'
                )
                _ = CartesianCombos(*my_fixture_input)
        else:
            print(
                f'My initialization input: {my_fixture_input} (Expect test pass!)'
            )
            _ = CartesianCombos(*my_fixture_input)

    @pytest.mark.parametrize(
        ('input_term_index', 'expected_fail'),
        [
            (i, x)
            for i, x in enumerate(
                [False, False, False, False, False, False, False, True, True]
            )
        ],
    )
    def test_single_set(self, input_term_index, expected_fail):
        """Tests the initialization of the Single set combos object.

        Parameters
        ----------
        input_term_index : int
            Index to retrieve the correct index for the test input fixture for the initialization of the 'CartesianCombos' object.
        expected_fail : bool
            True if the fail is expected, False if not.
        """
        # retrieve the correct fixture value
        my_fixture_input = self.input_list[input_term_index]  # type: ignore
        # initialize the object (requires iterable input)
        if expected_fail:
            with pytest.raises(Exception) as e_info:
                print(
                    f'My initialization input: {my_fixture_input} (Expect exception being raised and captured!)'
                )
                _ = SingleSetCombos(my_fixture_input)
        else:
            print(
                f'My initialization input: {my_fixture_input} (Expect test pass!)'
            )
            _ = SingleSetCombos(my_fixture_input)
