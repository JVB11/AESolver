"""Python pytest module for the gyre_cutter enumeration class.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import pytest
import json
import numpy as np

# import the gyre_cutter package functionality
from gyre_cutter import GYRECutter

# import utility classes for pytest
from pytest_util_classes import EVs  # type: ignore


# generate enumeration object for default test input values
class DefaultInputValues(EVs):
    """Contains the default test input values."""

    X = np.ones(10).tobytes()
    BRUNT_N2_PROFILE = np.ones(11).tobytes()
    CUT_LIST = json.dumps(['_brunt_N2_profile'])


# generate enumeration object for default test output values
class DefaultOutputValues(EVs):
    """Contains the default test output values."""

    BRUNT_N2_PROFILE = np.ones(10).tobytes()


# construct a pytest class fixture with the necessary attributes
@pytest.fixture(scope='class')
def setup_test_class(request):
    """Generates a fixture for the testing class that makes information available for testing."""
    # convert the stored enumeration values to actual numpy arrays and lists
    _x = DefaultInputValues.get_value('X')
    _bn2_i = DefaultInputValues.get_value('BRUNT_N2_PROFILE')
    _bn2_o = DefaultOutputValues.get_value('BRUNT_N2_PROFILE')
    _cl = DefaultInputValues.get_value('CUT_LIST')
    # import sys
    # sys.exit()
    # set the attributes necessary for the test
    request.cls._x = _x
    request.cls._M_r = _x
    request.cls._rho = _x
    request.cls._P = _x
    request.cls._Gamma_1 = _x
    request.cls._brunt_N2_profile = _bn2_i
    # set the attributes that need to be cut
    request.cls.cut_list = _cl
    # set the cut slice
    request.cls.cut_slice = np.s_[1:]
    # set the expected output value
    request.cls.brunt_out = _bn2_o


# generate the test class
@pytest.mark.usefixtures('setup_test_class')
class TestGyreCutter:
    """Python test class for the GYRE cutter package."""

    # attribute type declarations
    _x: np.ndarray
    _M_r: np.ndarray
    _rho: np.ndarray
    _P: np.ndarray
    _Gamma_1: np.ndarray
    _brunt_N2_profile: np.ndarray
    cut_list: list[str]
    cut_slice: np.s_  # type: ignore
    brunt_out: np.ndarray

    def test_correct_slice(self):
        """Tests if the GYRE cutter function is producing correct output."""
        # use the GYREcutter object to perform the slicing
        GYRECutter.cutter(self, self.cut_list, self.cut_slice)
        # get lengths of two different properties
        new_len: int = self._brunt_N2_profile.shape[0]
        result_len: int = self.brunt_out.shape[0]
        # assert that the cutter function is producing correct output
        assert all(
            [
                np.allclose(self.brunt_out, self._brunt_N2_profile),
                new_len == result_len,
            ]
        )
