"""Pytest module used to test the numerical differentiator module.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
import pytest
import numpy as np
from pytest_util_classes import EVs  # type: ignore
from num_deriv import NumericalDifferentiator


class InputVals(EVs):
    """Holds the input values for the tests."""

    X_VALS = np.linspace(1.0, 5.0, num=5, dtype=np.float64).tobytes()
    SQUARED = np.array([1.0, 4.0, 9.0, 16.0, 25.0], dtype=np.float64).tobytes()
    HOUGH = np.array(
        [0.45364445, 0.66539314, 0.75544452, 0.66539314, 0.45364445],
        dtype=np.float64,
    ).tobytes()


class OutputVals(EVs):
    """Holds the output values for the tests."""

    FD_SQUARED = (
        2.0 * np.linspace(1.0, 5.0, num=5, dtype=np.float64)
    ).tobytes()
    FD_HOUGH = np.array(
        [0.27259735, 0.15090003, 0.0, -0.15090003, -0.27259735],
        dtype=np.float64,
    ).tobytes()
    SD_SQUARED = (2.0 * np.ones((5,), dtype=np.float64)).tobytes()
    SD_HOUGH = np.array(
        [-0.10709595, -0.13629867, -0.15090003, -0.13629867, -0.10709595],
        dtype=np.float64,
    ).tobytes()


@pytest.fixture(scope='class', params=['squared', 'hough'])
def setup_test_class(request):
    """Generates a fixture for the testing class."""
    request.cls.input_vec = InputVals.get_value(request.param)
    request.cls.fd_output = OutputVals.get_value(f'fd_{request.param}')
    request.cls.sd_output = OutputVals.get_value(f'sd_{request.param}')
    request.cls.x_vals = InputVals.get_value('x_vals')
    request.cls.order_derivative = 2


@pytest.mark.usefixtures('setup_test_class')
class TestNumericalDifferentiator:
    """Python test class for the num_deriv package and NumericalDifferentiator object."""
    input_vec: np.ndarray
    fd_output: np.ndarray
    sd_output: np.ndarray
    x_vals: np.ndarray
    order_derivative: int

    def test_gradient_fd_defaults(self):
        my_obj = NumericalDifferentiator(
            order_derivative=self.order_derivative,
            differentiation_method='gradient',
        )
        my_deriv = my_obj.differentiate(
            self.x_vals, self.input_vec, edge_order=2
        )
        assert np.allclose(self.fd_output, my_deriv[0][1], equal_nan=True)

    def test_gradient_sd_defaults(self):
        my_obj = NumericalDifferentiator(
            order_derivative=self.order_derivative,
            differentiation_method='gradient',
        )
        my_deriv = my_obj.differentiate(
            self.x_vals, self.input_vec, edge_order=2
        )
        my_obj_second = NumericalDifferentiator(
            order_derivative=self.order_derivative,
            differentiation_method='gradient',
        )
        my_deriv_sec = my_obj_second.differentiate(
            self.x_vals, my_deriv[0][1], edge_order=2
        )
        assert np.allclose(self.sd_output, my_deriv_sec[0][1], equal_nan=True)

    def test_gradient_fd_lower_edge(self):
        my_obj = NumericalDifferentiator(
            order_derivative=self.order_derivative,
            differentiation_method='gradient',
        )
        my_deriv = my_obj.differentiate(
            self.x_vals, self.input_vec, edge_order=1
        )
        # DUE TO BOUNDARY CASES, rtol has to be set incredibly high!
        assert np.allclose(
            self.fd_output, my_deriv[0][1], rtol=0.5, equal_nan=True
        )
        assert np.allclose(
            self.fd_output[1:-1], my_deriv[0][1][1:-1], equal_nan=True
        )  # without edges = OK to normal numpy standards

    def test_gradient_sd_lower_edge(self):
        my_obj = NumericalDifferentiator(
            order_derivative=self.order_derivative,
            differentiation_method='gradient',
        )
        my_deriv = my_obj.differentiate(
            self.x_vals, self.input_vec, edge_order=1
        )
        my_obj_second = NumericalDifferentiator(
            order_derivative=self.order_derivative,
            differentiation_method='gradient',
        )
        my_deriv_sec = my_obj_second.differentiate(
            self.x_vals, my_deriv[0][1], edge_order=1
        )
        # perform assertion for baseline model: need 100% differences
        # WITH EDGES
        assert np.allclose(
            self.sd_output, my_deriv_sec[0][1], rtol=1.0, equal_nan=True
        )
        # NO EDGES (HERE: the first and last two elements are an edge!)
        assert np.allclose(
            self.sd_output[2:-2], my_deriv_sec[0][1][2:-2], equal_nan=True
        )  # OK to normal numpy standards!

    def test_fornberg_interpolation_defaults_fd(self):
        """Tests the fornberg interpolation method's first order derivative."""
        my_obj = NumericalDifferentiator(
            order_derivative=self.order_derivative,
            differentiation_method='fornberg',
        )
        my_deriv = my_obj.differentiate(
            self.x_vals,
            self.input_vec,
            original_formulation=False,
            interpolation_order=4,
        )
        # assert that all values of the derivatives are close to within 
        # an absolute value of 0.05 only --> EDGES ARE INCLUDED
        assert np.allclose(
            self.fd_output, my_deriv[0][1], atol=0.05, equal_nan=True
        )
        # WITHOUT EDGES, it is fine to within 15% -> not great
        assert np.allclose(
            self.fd_output[1:-1],
            my_deriv[0][1][1:-1],
            rtol=0.15,
            equal_nan=True,
        )

    def test_fornberg_interpolation_defaults_sd(self):
        """Tests the fornberg interpolation method's second-order derivative."""
        my_obj = NumericalDifferentiator(
            order_derivative=self.order_derivative,
            differentiation_method='fornberg',
        )
        my_deriv = my_obj.differentiate(
            self.x_vals,
            self.input_vec,
            original_formulation=False,
            interpolation_order=4,
        )
        # assert that all values of the derivatives are close to within 
        # an absolute value of 0.2 only --> EDGES ARE INCLUDED
        assert np.allclose(
            self.sd_output, my_deriv[0][2], atol=0.2, equal_nan=True
        )
        # WITHOUT EDGES, it is fine to within 21% -> not great
        assert np.allclose(
            self.sd_output[1:-1],
            my_deriv[0][2][1:-1],
            rtol=0.21,
            equal_nan=True,
        )

    def test_gradient_fd_defaults_interp(self):
        my_obj = NumericalDifferentiator(
            order_derivative=self.order_derivative,
            differentiation_method='gradient',
            perform_interpolation=True,
            interpolation_factor=10,
        )
        my_deriv = my_obj.differentiate(
            self.x_vals, self.input_vec, edge_order=2
        )
        # perform assertion for baseline model --> atol needs to be set to 1.0 to enclose EDGES!
        assert np.allclose(
            self.fd_output, my_deriv[0][1], equal_nan=True, atol=1.0
        )
        # EDGES ARE IMPORTANT --> same result as without interpolation if we do not consider the edges!!!
        assert np.allclose(
            self.fd_output[1:-1], my_deriv[0][1][1:-1], equal_nan=True
        )

    def test_gradient_sd_defaults_interp(self):
        my_obj = NumericalDifferentiator(
            order_derivative=self.order_derivative,
            differentiation_method='gradient',
            perform_interpolation=True,
            interpolation_factor=10,
        )
        my_deriv = my_obj.differentiate(
            self.x_vals, self.input_vec, edge_order=2
        )
        my_obj_second = NumericalDifferentiator(
            order_derivative=self.order_derivative,
            differentiation_method='gradient',
            perform_interpolation=True,
            interpolation_factor=10,
        )
        my_deriv_sec = my_obj_second.differentiate(
            self.x_vals, my_deriv[0][1], edge_order=2
        )
        # EDGES ARE IMPORTANT --> need atol of 1.0 (50% of actual value!)
        assert np.allclose(
            self.sd_output, my_deriv_sec[0][1], equal_nan=True, atol=1.0
        )
        # without edges - it is ok!
        assert np.allclose(
            self.sd_output[2:-2], my_deriv_sec[0][1][2:-2], equal_nan=True
        )

    def test_gradient_fd_lower_edge_interp(self):
        my_obj = NumericalDifferentiator(
            order_derivative=self.order_derivative,
            differentiation_method='gradient',
            perform_interpolation=True,
            interpolation_factor=10,
        )
        my_deriv = my_obj.differentiate(
            self.x_vals, self.input_vec, edge_order=1
        )
        # perform assertion, DUE TO BOUNDARY CASES, rtol has to be set incredibly high 
        # --> HOWEVER, it is less high than when interpolation is NOT used!!!
        assert np.allclose(
            self.fd_output, my_deriv[0][1], rtol=0.4, equal_nan=True
        )
        assert np.allclose(
            self.fd_output[1:-1], my_deriv[0][1][1:-1], equal_nan=True
        )  # without edges = OK to normal numpy standards

    def test_gradient_sd_lower_edge_interp(self):
        my_obj = NumericalDifferentiator(
            order_derivative=self.order_derivative,
            differentiation_method='gradient',
            perform_interpolation=True,
            interpolation_factor=10,
        )
        my_deriv = my_obj.differentiate(
            self.x_vals, self.input_vec, edge_order=1
        )
        my_obj_second = NumericalDifferentiator(
            order_derivative=self.order_derivative,
            differentiation_method='gradient',
            perform_interpolation=True,
            interpolation_factor=10,
        )
        my_deriv_sec = my_obj_second.differentiate(
            self.x_vals, my_deriv[0][1], edge_order=1
        )
        # perform assertion for baseline model: need 100% differences
        # WITH EDGES
        assert np.allclose(
            self.sd_output, my_deriv_sec[0][1], rtol=1.0, equal_nan=True
        )
        # NO EDGES (HERE: the first and last two elements are an edge!)
        assert np.allclose(
            self.sd_output[2:-2], my_deriv_sec[0][1][2:-2], equal_nan=True
        )  # OK to normal numpy standards!

    def test_fornberg_interpolation_defaults_fd_interp(self):
        """Tests the fornberg interpolation method's first order derivative."""
        my_obj = NumericalDifferentiator(
            order_derivative=self.order_derivative,
            differentiation_method='fornberg',
            perform_interpolation=True,
            interpolation_factor=10,
        )
        my_deriv = my_obj.differentiate(
            self.x_vals,
            self.input_vec,
            original_formulation=False,
            interpolation_order=4,
        )
        # assert that all values of the derivatives are close
        # --> EDGES are still important!
        assert np.allclose(
            self.fd_output, my_deriv[0][1], equal_nan=True, atol=1.0
        )
        # without edges --> all OK
        assert np.allclose(
            self.fd_output[1:-1], my_deriv[0][1][1:-1], equal_nan=True
        )

    def test_fornberg_interpolation_defaults_sd_interp(self):
        """Tests the fornberg interpolation method's second-order derivative."""
        my_obj = NumericalDifferentiator(
            order_derivative=self.order_derivative,
            differentiation_method='fornberg',
            perform_interpolation=True,
            interpolation_factor=10,
        )
        my_deriv = my_obj.differentiate(
            self.x_vals,
            self.input_vec,
            original_formulation=False,
            interpolation_order=4,
        )
        my_obj2 = NumericalDifferentiator(
            order_derivative=self.order_derivative,
            differentiation_method='fornberg',
            perform_interpolation=True,
            interpolation_factor=10,
        )
        my_deriv2 = my_obj2.differentiate(
            self.x_vals,
            my_deriv[0][1],
            original_formulation=False,
            interpolation_order=4,
        )
        # assert that all values of the derivatives are close to within 
        # an absolute value of 1.0 or 0.2 only --> EDGES ARE INCLUDED
        assert np.allclose(
            self.sd_output, my_deriv2[0][1], atol=1.0, equal_nan=True
        )
        # WITHOUT EDGES --> NO ISSUES
        assert np.allclose(
            self.sd_output[2:-2], my_deriv2[0][1][2:-2], equal_nan=True
        )

    def test_fornberg_interpolation_increase_order_sd_interp(self):
        """Tests the fornberg interpolation method's second-order derivative."""
        my_obj = NumericalDifferentiator(
            order_derivative=self.order_derivative,
            differentiation_method='fornberg',
            perform_interpolation=True,
            interpolation_factor=10,
        )
        my_deriv = my_obj.differentiate(
            self.x_vals,
            self.input_vec,
            original_formulation=False,
            interpolation_order=6,
        )
        my_obj2 = NumericalDifferentiator(
            order_derivative=self.order_derivative,
            differentiation_method='fornberg',
            perform_interpolation=True,
            interpolation_factor=10,
        )
        my_deriv2 = my_obj2.differentiate(
            self.x_vals,
            my_deriv[0][1],
            original_formulation=False,
            interpolation_order=6,
        )
        # assert that all values of the derivatives are close to 
        # within an absolute value of 1.0 only --> EDGES ARE INCLUDED
        assert np.allclose(
            self.sd_output, my_deriv2[0][1], atol=1.0, equal_nan=True
        )
        # WITHOUT EDGES --> NO ISSUES
        assert np.allclose(
            self.sd_output[2:-2], my_deriv2[0][1][2:-2], equal_nan=True
        )
