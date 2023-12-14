"""Pytest module used to test the 'carr_conv' module.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import pytest
import numpy as np

# import custom module for pytest generation
from pytest_util_classes import EVs  # type: ignore

# import module functions
from carr_conv import re_im, re_im_parts


# generate enumeration object for default test input values
class InputVals(EVs):
    """Holds the input values for the tests."""

    RE_IM = np.array(
        np.ones(10), dtype=[('re', np.float64), ('im', np.float64)]
    ).tobytes()
    RE_IM_REAL = np.ones(10, dtype=np.float64).tobytes()
    RE_IM_IMAG = np.ones(10, dtype=np.float64).tobytes()
    DIM_EXPR = 5.0


# generate enumeration object for default test output values
class OutputVals(EVs):
    """Holds the output values for the tests."""

    RE_IM = ((5.0 + 5.0j) * np.ones(10, dtype=np.complex128)).tobytes()
    RE_IM_PARTS = ((1.0 + 1.0j) * np.ones(10, dtype=np.complex128)).tobytes()


# construct a pytest class fixture with the necessary attributes
@pytest.fixture(scope='class')
def setup_test_class(request):
    """Generates a fixture for the testing class."""
    # convert stored bytes objects to array views with specific types
    _re_im_i = InputVals.get_value(
        'RE_IM', dtype_buffered=[('re', np.float64), ('im', np.float64)]
    )
    _re_im_o = OutputVals.get_value('RE_IM', dtype_buffered=np.complex128)
    _re_im_p_o = OutputVals.get_value(
        'RE_IM_PARTS', dtype_buffered=np.complex128
    )
    # set attributes for test
    request.cls.parts_real = InputVals.get_value('RE_IM_REAL')
    request.cls.parts_imag = InputVals.get_value('RE_IM_IMAG')
    request.cls.parts_out = _re_im_p_o
    request.cls.re_im_in = _re_im_i
    request.cls.re_im_out = _re_im_o
    request.cls.dim_expr = InputVals.get_value('DIM_EXPR')


# generate the test class
@pytest.mark.usefixtures('setup_test_class')
class TestCarrConv:
    """Python test class for the carr_conv package."""

    # attribute type declarations
    parts_real: np.ndarray
    parts_imag: np.ndarray
    parts_out: np.ndarray
    re_im_in: np.ndarray
    re_im_out: np.ndarray
    dim_expr: float

    def test_re_im(self):
        """Tests functionality of the 're_im' function of carr_conv."""
        # use the re_im function to generate output
        my_complex_arr = re_im(self.re_im_in, self.dim_expr)
        # compare with expected output
        assert np.allclose(my_complex_arr, self.re_im_out)

    def test_re_im_parts(self):
        """Tests functionality of the 're_im_parts' function of carr_conv."""
        # use the re_im_parts function to generate output
        my_complex_arr = re_im_parts(self.parts_real, self.parts_imag)
        # compare with expected output
        assert np.allclose(my_complex_arr, self.parts_out)
