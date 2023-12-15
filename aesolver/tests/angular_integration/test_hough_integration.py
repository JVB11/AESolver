"""Pytest module that tests the angular integration methods used to integrate the Hough functions.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import pytest
import numpy as np
import math
import typing
from scipy.interpolate import interp1d

# import pytest helper class
from pytest_util_classes import EVs

# import local helper modules
from mock_input_enum import InputDataEnum
from data_object_mock import MockDataObj
from input_params import InputValues
from input_params import (
    InputParametersNoAlternativeDivSinProcedure as InputParamsNoAlt,
)
from output_params import ExpectedOutputParameters

# import module to be tested
from aesolver.angular_integration.rotating_tar_angular import HI


# define a test mu value array
_mu_val_arr = np.linspace(
    -0.999999999999999, 0.999999999999999, num=200, dtype=np.float64
)
_theta_val_arr = np.arccos(_mu_val_arr)


# define the input data for stringent angular integral tests
class InputDataZeroZeros(InputDataEnum):
    """Contains the input data for integrations of data arrays containing ones: simulates integrals over P_0^0 functions."""

    # define enumeration attributes
    MY_DATA_ARR1 = np.ones_like(_mu_val_arr).tobytes()
    MY_DATA_ARR2 = np.ones_like(_mu_val_arr).tobytes()
    MY_DATA_ARR3 = np.ones_like(_mu_val_arr).tobytes()
    MU_VALUES = _mu_val_arr.tobytes()
    THETA_VALUES = _theta_val_arr.tobytes()


class InputDataOneZeros(InputDataEnum):
    """Contains the input data for integrations of data arrays containing ones: simulates integrals over P_1^0 functions."""

    # define enumeration attributes
    MY_DATA_ARR1 = _mu_val_arr.tobytes()
    MY_DATA_ARR2 = _mu_val_arr.tobytes()
    MY_DATA_ARR3 = _mu_val_arr.tobytes()
    MU_VALUES = _mu_val_arr.tobytes()
    THETA_VALUES = _theta_val_arr.tobytes()


class InputDataOneOnes(InputDataEnum):
    """Contains the input data for integrations of data arrays containing ones: simulates integrals over P_1^1 functions."""

    # define enumeration attributes
    MY_DATA_ARR1 = np.sqrt(1.0 - _mu_val_arr**2.0).tobytes()
    MY_DATA_ARR2 = np.sqrt(1.0 - _mu_val_arr**2.0).tobytes()
    MY_DATA_ARR3 = np.sqrt(1.0 - _mu_val_arr**2.0).tobytes()
    MU_VALUES = _mu_val_arr.tobytes()
    THETA_VALUES = _theta_val_arr.tobytes()


class InputDataTwoMinOnes(InputDataEnum):
    """Contains the input data for integrations of data arrays containing ones: simulates integrals over P_2^-1 functions."""

    # define enumeration attributes
    MY_DATA_ARR1 = (np.sqrt(1.0 - _mu_val_arr**2.0) * _mu_val_arr).tobytes()
    MY_DATA_ARR2 = (np.sqrt(1.0 - _mu_val_arr**2.0) * _mu_val_arr).tobytes()
    MY_DATA_ARR3 = (np.sqrt(1.0 - _mu_val_arr**2.0) * _mu_val_arr).tobytes()
    MU_VALUES = _mu_val_arr.tobytes()
    THETA_VALUES = _theta_val_arr.tobytes()


class InputDataTwoTwos(InputDataEnum):
    """Contains the input data for integrations of data arrays containing ones: simulates integrals over P_2^2 functions."""

    # define enumeration attributes
    MY_DATA_ARR1 = (1.0 - _mu_val_arr**2.0).tobytes()
    MY_DATA_ARR2 = (1.0 - _mu_val_arr**2.0).tobytes()
    MY_DATA_ARR3 = (1.0 - _mu_val_arr**2.0).tobytes()
    MU_VALUES = _mu_val_arr.tobytes()
    THETA_VALUES = _theta_val_arr.tobytes()


class InputDataTwoZeros(InputDataEnum):
    """Contains the input data for integrations of data arrays containing ones: simulates integrals over P_2^0 functions."""

    # define enumeration attributes
    MY_DATA_ARR1 = (3.0 * ((_mu_val_arr) ** 2.0) - 1.0).tobytes()
    MY_DATA_ARR2 = (3.0 * ((_mu_val_arr) ** 2.0) - 1.0).tobytes()
    MY_DATA_ARR3 = (3.0 * ((_mu_val_arr) ** 2.0) - 1.0).tobytes()
    MU_VALUES = _mu_val_arr.tobytes()
    THETA_VALUES = _theta_val_arr.tobytes()


class InputDataTwoTwoOneOnes(InputDataEnum):
    """Contains the input data for integrations of data arrays containing ones: simulates integrals over P_2^2 * P_1^1 * P_1^1."""

    # define enumeration attributes
    MY_DATA_ARR1 = (1.0 - _mu_val_arr**2.0).tobytes()
    MY_DATA_ARR2 = np.sqrt(1.0 - _mu_val_arr**2.0).tobytes()
    MY_DATA_ARR3 = np.sqrt(1.0 - _mu_val_arr**2.0).tobytes()
    MU_VALUES = _mu_val_arr.tobytes()
    THETA_VALUES = _theta_val_arr.tobytes()


class InputDataTwoTwoTwoOnes(InputDataEnum):
    """Contains the input data for integrations of data arrays containing ones: simulates integrals over P_2^2 * P_2^1 * P_2^1."""

    # define enumeration attributes
    MY_DATA_ARR1 = (1.0 - _mu_val_arr**2.0).tobytes()
    MY_DATA_ARR2 = ((_mu_val_arr) * np.sqrt(1.0 - _mu_val_arr**2.0)).tobytes()
    MY_DATA_ARR3 = ((_mu_val_arr) * np.sqrt(1.0 - _mu_val_arr**2.0)).tobytes()
    MU_VALUES = _mu_val_arr.tobytes()
    THETA_VALUES = _theta_val_arr.tobytes()


class InputDataTwoTwoTwoZeros(InputDataEnum):
    """Contains the input data for integrations of data arrays containing ones: simulates integrals over P_2^2 * P_2^0 * P_2^0."""

    # define enumeration attributes
    MY_DATA_ARR1 = (1.0 - _mu_val_arr**2.0).tobytes()
    MY_DATA_ARR2 = (3.0 * ((_mu_val_arr) ** 2.0) - 1.0).tobytes()
    MY_DATA_ARR3 = (3.0 * ((_mu_val_arr) ** 2.0) - 1.0).tobytes()
    MU_VALUES = _mu_val_arr.tobytes()
    THETA_VALUES = _theta_val_arr.tobytes()


# generate a fixture used to set up the test class
@pytest.fixture(
    scope='class',
    params=[  # FLOATING POINT ACCURACY NUMPY IS 1e-15!
        # ABSOLUTE ACCURACIES:
        # first = trapz, no div_sin
        # second = trapz, div_sin
        # third = CG-quad, no div_sin
        # fourth = CG-quad, div_sin
        # fifth = BLOW-UP protection by CG-quad, otherwise trapz
        (
            InputDataZeroZeros,
            (2.0, np.pi),
            (2.0, np.pi),
            [0, 0, 0],
            (1e-15, 1e7, 1e-1, 1e-15, 1e-15, 1e-15),
        ),
        (
            InputDataOneZeros,
            (0.0, 0.0),
            (0.0, 0.0),
            [0, 0, 0],
            (1e-15, 1e-15, 1e-15, 1e-15, 1e-15, 1e-15),
        ),
        (
            InputDataOneOnes,
            ((3.0 * np.pi) / 8.0, 4.0 / 3.0),
            (0.0, 0.0),
            [1, 1, 1],
            (1e-4, 1e-3, 1e-3, 1e-3, 1e-4, 1e-3),
        ),
        (
            InputDataTwoMinOnes,
            (0.0, 0.0),
            (0.0, 0.0),
            [-1, -1, -1],
            (1e-15, 1e-15, 1e-15, 1e-15, 1e-15, 1e-15),
        ),
        (
            InputDataTwoTwos,
            (32.0 / 35.0, (5.0 * np.pi) / 16.0),
            (0.0, 0.0),
            [2, 2, 2],
            (1e-8, 1e-7, 1e-3, 1e-3, 1e-8, 1e-7),
        ),
        (
            InputDataTwoZeros,
            (32.0 / 35.0, (29.0 * np.pi) / 16.0),
            (32.0 / 35.0, (29.0 * np.pi) / 16.0),
            [0, 0, 0],
            (1e-2, 1e8, 4e-1, 1e-1, 1e-2, 1e-1),
        ),
        (
            InputDataTwoTwoOneOnes,
            (16.0 / 15.0, (3.0 * np.pi) / 8.0),
            (16.0 / 15.0, (3.0 * np.pi) / 8.0),
            [2, 1, 1],
            (1e-15, 1e-5, 1e-3, 1e-3, 1e-15, 1e-5),
        ),
        (
            InputDataTwoTwoTwoOnes,
            (16.0 / 105.0, np.pi / 16.0),
            (16.0 / 105.0, np.pi / 16.0),
            [2, 1, 1],
            (1e-7, 1e-5, 1e-4, 1e-3, 1e-7, 1e-5),
        ),
        (
            InputDataTwoTwoTwoZeros,
            (16.0 / 21.0, (5.0 * np.pi) / 16.0),
            (0.0, 0.0),
            [2, 0, 0],
            (1e-3, 1e-1, 1e-2, 1e-2, 1e-3, 1e-1),
        ),
    ],  # ONLY USE CG-QUAD FOR BLOW-UP???
)
def spherical_har_test_setup(request) -> None:
    """Generates a fixture for the test class for spherical harmonic integrations."""
    # 'SPHERICAL HARMONIC' INTEGRALS FOR THOROUGH TESTING
    request.cls.data_class: MockDataObj = MockDataObj(request.param[0])
    # expected output parameters
    request.cls.expect_pure_theta: tuple[float] = request.param[1]
    request.cls.expect_full: tuple[float] = request.param[2]
    # input parameters
    request.cls.input_params: dict[str, typing.Any] = {
        'descrips': [f'my_data_arr{x}' for x in range(1, 4)],
        'conj': (True, False, False),
        'mul_with_mu': False,
        'm_list_for_azimuthal_check': request.param[3],
        'nr_omp_threads': 1,
        'use_parallel': False,
    }
    # --> NEED TO SET 'div_by_sin' separately!
    # accuracy of comparison
    request.cls.accuracy: float = request.param[4]


# define test class
@pytest.mark.usefixtures('spherical_har_test_setup')
class TestSphericalHarmonicThetas:
    """Tests whether the angular integration methods perform OK for simplified spherical harmonics (non-normalized)."""

    # attribute type declarations
    data_class: MockDataObj
    expect_pure_theta: tuple[float]
    expect_full: tuple[float]
    input_params: dict[str, typing.Any]
    accuracy: float

    def test_only_theta(self):
        """Tests only the theta integration, using mu variables."""
        print(self.input_params['m_list_for_azimuthal_check'])
        # angular integral using module
        my_result = HI.quadratic_angular_integral(
            my_obj=self.data_class,
            only_theta=True,
            descrips=self.input_params['descrips'],
            conj=self.input_params['conj'],
            div_by_sin=False,
            mul_with_mu=self.input_params['mul_with_mu'],
            m_list_for_azimuthal_check=self.input_params[
                'm_list_for_azimuthal_check'
            ],
            nr_omp_threads=self.input_params['nr_omp_threads'],
            use_parallel=self.input_params['use_parallel'],
            use_cheby_integration=False,
        )
        # assert if the result is as expected
        assert math.isclose(
            my_result,
            2.0 * np.pi * self.expect_pure_theta[0],
            abs_tol=self.accuracy[0],
        )

    def test_only_theta_cheby(self):
        """Tests only the theta integration, using Chebyshev-Gauss quadrature."""
        # angular integral using module
        my_result = HI.quadratic_angular_integral(
            my_obj=self.data_class,
            only_theta=True,
            descrips=self.input_params['descrips'],
            conj=self.input_params['conj'],
            div_by_sin=False,
            mul_with_mu=self.input_params['mul_with_mu'],
            m_list_for_azimuthal_check=self.input_params[
                'm_list_for_azimuthal_check'
            ],
            nr_omp_threads=self.input_params['nr_omp_threads'],
            use_parallel=self.input_params['use_parallel'],
            use_cheby_integration=True,
        )
        # assert if the result is as expected
        assert math.isclose(
            my_result,
            2.0 * np.pi * self.expect_pure_theta[0],
            abs_tol=self.accuracy[2],
        )

    def test_only_theta_div_by_sin(self):
        """Tests only the theta integration for div_by_sin integrals, using mu variables."""
        # angular integral using module
        my_result = HI.quadratic_angular_integral(
            my_obj=self.data_class,
            only_theta=True,
            descrips=self.input_params['descrips'],
            conj=self.input_params['conj'],
            div_by_sin=True,
            mul_with_mu=self.input_params['mul_with_mu'],
            m_list_for_azimuthal_check=self.input_params[
                'm_list_for_azimuthal_check'
            ],
            nr_omp_threads=self.input_params['nr_omp_threads'],
            use_parallel=self.input_params['use_parallel'],
            use_cheby_integration=False,
        )
        # assert if the result is as expected
        assert math.isclose(
            my_result,
            2.0 * np.pi * self.expect_pure_theta[1],
            abs_tol=self.accuracy[1],
        )

    def test_only_theta_div_by_sin_cheby(self):
        """Tests only the theta integration for div_by_sin integrals using Chebyshev-Gauss quadrature."""
        # angular integral using module
        my_result = HI.quadratic_angular_integral(
            my_obj=self.data_class,
            only_theta=True,
            descrips=self.input_params['descrips'],
            conj=self.input_params['conj'],
            div_by_sin=True,
            mul_with_mu=self.input_params['mul_with_mu'],
            m_list_for_azimuthal_check=self.input_params[
                'm_list_for_azimuthal_check'
            ],
            nr_omp_threads=self.input_params['nr_omp_threads'],
            use_parallel=self.input_params['use_parallel'],
            use_cheby_integration=True,
        )
        # assert if the result is as expected
        assert math.isclose(
            my_result,
            2.0 * np.pi * self.expect_pure_theta[1],
            abs_tol=self.accuracy[3],
        )

    def test_only_theta_blow_up_protection(self):
        """Tests only the theta integration using Chebyshev-Gauss quadrature only when blowing up."""
        # angular integral using module
        my_result = HI.quadratic_angular_integral(
            my_obj=self.data_class,
            only_theta=True,
            descrips=self.input_params['descrips'],
            conj=self.input_params['conj'],
            div_by_sin=False,
            mul_with_mu=self.input_params['mul_with_mu'],
            m_list_for_azimuthal_check=self.input_params[
                'm_list_for_azimuthal_check'
            ],
            nr_omp_threads=self.input_params['nr_omp_threads'],
            use_parallel=self.input_params['use_parallel'],
            use_cheby_integration=False,
            cheby_blow_up_protection=True,
        )
        # assert if the result is as expected
        assert math.isclose(
            my_result,
            2.0 * np.pi * self.expect_pure_theta[0],
            abs_tol=self.accuracy[4],
        )

    def test_only_theta_div_by_sin_blow_up_protection(self):
        """Tests only the theta integration for div_by_sin integrals using Chebyshev-Gauss quadrature only when blowing up."""
        # angular integral using module
        my_result = HI.quadratic_angular_integral(
            my_obj=self.data_class,
            only_theta=True,
            descrips=self.input_params['descrips'],
            conj=self.input_params['conj'],
            div_by_sin=True,
            mul_with_mu=self.input_params['mul_with_mu'],
            m_list_for_azimuthal_check=self.input_params[
                'm_list_for_azimuthal_check'
            ],
            nr_omp_threads=self.input_params['nr_omp_threads'],
            use_parallel=self.input_params['use_parallel'],
            use_cheby_integration=False,
            cheby_blow_up_protection=True,
        )
        # assert if the result is as expected
        assert math.isclose(
            my_result,
            2.0 * np.pi * self.expect_pure_theta[1],
            abs_tol=self.accuracy[5],
        )

    def test_full(self):
        """Tests the theta and phi integration."""
        # angular integral using module
        my_result = HI.quadratic_angular_integral(
            my_obj=self.data_class,
            descrips=self.input_params['descrips'],
            conj=self.input_params['conj'],
            div_by_sin=False,
            mul_with_mu=self.input_params['mul_with_mu'],
            m_list_for_azimuthal_check=self.input_params[
                'm_list_for_azimuthal_check'
            ],
            nr_omp_threads=self.input_params['nr_omp_threads'],
            use_parallel=self.input_params['use_parallel'],
        )
        # assert if the result is as expected
        assert math.isclose(
            my_result,
            2.0 * np.pi * self.expect_full[0],
            abs_tol=self.accuracy[0],
        )

    def test_full_div_by_sin(self):
        """Tests the theta and phi integration for div_by_sin integrals."""
        # angular integral using module
        my_result = HI.quadratic_angular_integral(
            my_obj=self.data_class,
            descrips=self.input_params['descrips'],
            conj=self.input_params['conj'],
            div_by_sin=True,
            mul_with_mu=self.input_params['mul_with_mu'],
            m_list_for_azimuthal_check=self.input_params[
                'm_list_for_azimuthal_check'
            ],
            nr_omp_threads=self.input_params['nr_omp_threads'],
            use_parallel=self.input_params['use_parallel'],
        )
        # assert if the result is as expected
        assert math.isclose(
            my_result,
            2.0 * np.pi * self.expect_full[1],
            abs_tol=self.accuracy[1],
        )

    def test_full_cheby(self):
        """Tests the theta and phi integration."""
        # angular integral using module
        my_result = HI.quadratic_angular_integral(
            my_obj=self.data_class,
            descrips=self.input_params['descrips'],
            conj=self.input_params['conj'],
            div_by_sin=False,
            mul_with_mu=self.input_params['mul_with_mu'],
            m_list_for_azimuthal_check=self.input_params[
                'm_list_for_azimuthal_check'
            ],
            nr_omp_threads=self.input_params['nr_omp_threads'],
            use_parallel=self.input_params['use_parallel'],
            use_cheby_integration=True,
            cheby_blow_up_protection=False,
        )
        # assert if the result is as expected
        assert math.isclose(
            my_result,
            2.0 * np.pi * self.expect_full[0],
            abs_tol=self.accuracy[2],
        )

    def test_full_div_by_sin_cheby(self):
        """Tests the theta and phi integration for div_by_sin integrals. """
        # angular integral using module
        my_result = HI.quadratic_angular_integral(
            my_obj=self.data_class,
            descrips=self.input_params['descrips'],
            conj=self.input_params['conj'],
            div_by_sin=True,
            mul_with_mu=self.input_params['mul_with_mu'],
            m_list_for_azimuthal_check=self.input_params[
                'm_list_for_azimuthal_check'
            ],
            nr_omp_threads=self.input_params['nr_omp_threads'],
            use_parallel=self.input_params['use_parallel'],
            use_cheby_integration=True,
            cheby_blow_up_protection=False,
        )
        # assert if the result is as expected
        assert math.isclose(
            my_result,
            2.0 * np.pi * self.expect_full[1],
            abs_tol=self.accuracy[3],
        )

    def test_full_blow_up_protection(self):
        """Tests the theta and phi integration."""
        # angular integral using module
        my_result = HI.quadratic_angular_integral(
            my_obj=self.data_class,
            descrips=self.input_params['descrips'],
            conj=self.input_params['conj'],
            div_by_sin=False,
            mul_with_mu=self.input_params['mul_with_mu'],
            m_list_for_azimuthal_check=self.input_params[
                'm_list_for_azimuthal_check'
            ],
            nr_omp_threads=self.input_params['nr_omp_threads'],
            use_parallel=self.input_params['use_parallel'],
            use_cheby_integration=False,
            cheby_blow_up_protection=True,
        )
        # assert if the result is as expected
        assert math.isclose(
            my_result,
            2.0 * np.pi * self.expect_full[0],
            abs_tol=self.accuracy[4],
        )

    def test_full_div_by_sin_blow_up_protection(self):
        """Tests the theta and phi integration for div_by_sin integrals. """
        # angular integral using module
        my_result = HI.quadratic_angular_integral(
            my_obj=self.data_class,
            descrips=self.input_params['descrips'],
            conj=self.input_params['conj'],
            div_by_sin=True,
            mul_with_mu=self.input_params['mul_with_mu'],
            m_list_for_azimuthal_check=self.input_params[
                'm_list_for_azimuthal_check'
            ],
            nr_omp_threads=self.input_params['nr_omp_threads'],
            use_parallel=self.input_params['use_parallel'],
            use_cheby_integration=False,
            cheby_blow_up_protection=True,
        )
        # assert if the result is as expected
        assert math.isclose(
            my_result,
            2.0 * np.pi * self.expect_full[1],
            abs_tol=self.accuracy[5],
        )


# define the input data for our tests
class InputDataExample(InputDataEnum):
    """Contains the input data for our tests.

    Notes
    -----
    Watch out with mu_values boundaries! If taken at -1 and 1, the corresponding functionality 'div_sin' stops working (returns NaNs because one is dividing by ZERO)!
    """

    # define enumeration attributes
    MY_DATA_ARR1 = np.linspace(0.3, 0.5, num=10000, dtype=np.float64).tobytes()
    MY_DATA_ARR2 = np.linspace(-0.1, 0.6, num=10000, dtype=np.float64).tobytes()
    MY_DATA_ARR3 = np.linspace(-0.5, 0.5, num=10000, dtype=np.float64).tobytes()
    MU_VALUES = np.linspace(
        -0.99999999, 0.99999999, num=10000, dtype=np.float64
    ).tobytes()


# generate the fixture used to set up the test class
@pytest.fixture(
    scope='class', params=InputParamsNoAlt.get_all_pytest_parameters()
)
def setup_test_class(request) -> None:
    """Generates a fixture that stores test class variables."""
    # set the data class variables
    request.cls.data_class: MockDataObj = MockDataObj(InputDataExample)
    # get the input data
    request.cls.input_params: dict[str, typing.Any] = InputValues.get_value(
        request.param
    )
    # get the expected output data
    request.cls.expected_output: float = ExpectedOutputParameters.get_value(
        request.param
    )
    # set the expected output accuracy
    # NOTE: LOWER ACCURACY SET FOR DIV_SIN APPLICATIONS
    request.cls.accuracy: float = 1e-3 if 'DIV_SIN' in request.param else 1e-5


# generate the test class
@pytest.mark.usefixtures('setup_test_class')
class TestHoughIntegration:
    """Test class for the HoughIntegration module of aesolver."""

    # attribute type declarations
    data_class: MockDataObj
    input_params: dict[str, typing.Any]
    input_params: dict[str, typing.Any]
    expected_output: float
    accuracy: float

    def test_hough_integration(self) -> None:
        # use the HoughIntegration module to compute
        # an angular integral
        my_result = HI.quadratic_angular_integral(
            my_obj=self.data_class,
            descrips=self.input_params['descrips'],
            conj=self.input_params['conj'],
            div_by_sin=self.input_params['div_by_sin'],
            mul_with_mu=self.input_params['mul_with_mu'],
            m_list_for_azimuthal_check=self.input_params[
                'm_list_for_azimuthal_check'
            ],
            nr_omp_threads=self.input_params['nr_omp_threads'],
            use_parallel=self.input_params['use_parallel'],
        )
        # assert if the result is as expected
        assert math.isclose(
            my_result, self.expected_output, abs_tol=self.accuracy
        )


# generate the fixture used to set up the test class
@pytest.fixture(
    scope='class', params=InputParamsNoAlt.get_all_pytest_parameters()
)
def setup_test_class_no_alt(request) -> None:
    """Generates a fixture that stores test class variables."""
    # set the data class variable
    request.cls.data_class: MockDataObj = MockDataObj(InputDataExample)
    # get the input data
    request.cls.input_params: dict[str, typing.Any] = InputValues.get_value(
        request.param
    )
    # get the expected output data
    request.cls.expected_output: float = ExpectedOutputParameters.get_value(
        request.param
    )
    # set the expected output accuracy
    # NOTE: LOWER ACCURACY SET FOR DIV_SIN APPLICATIONS
    request.cls.accuracy: float = 1e-3 if 'DIV_SIN' in request.param else 1e-5


# generate the test class
@pytest.mark.usefixtures('setup_test_class_no_alt')
class TestHoughIntegrationNoAltDivSin:
    """Test class for the HoughIntegration module of aesolver."""

    # attribute type declarations
    data_class: MockDataObj
    input_params: dict[str, typing.Any]
    input_params: dict[str, typing.Any]
    expected_output: float
    accuracy: float

    def test_hough_integration(self) -> None:
        """Tests the Hough integration methods without making use of the alternative DIV_SIN procedure!"""
        # use the HoughIntegration module to compute
        # an angular integral
        my_result = HI.quadratic_angular_integral(
            my_obj=self.data_class,
            descrips=self.input_params['descrips'],
            conj=self.input_params['conj'],
            div_by_sin=self.input_params['div_by_sin'],
            mul_with_mu=self.input_params['mul_with_mu'],
            m_list_for_azimuthal_check=self.input_params[
                'm_list_for_azimuthal_check'
            ],
            nr_omp_threads=self.input_params['nr_omp_threads'],
            use_parallel=self.input_params['use_parallel'],
            use_cheby_integration=False,
            cheby_blow_up_protection=False,
        )
        # assert if the result is as expected
        assert math.isclose(
            my_result, self.expected_output, abs_tol=self.accuracy
        )


class InputDataExampleChebyshevGauss(EVs):
    """Contains the input data for our tests of the Chebyshev-Gauss quadrature.

    Notes
    -----
    Watch out with mu_values boundaries! If taken at -1 and 1, the corresponding functionality 'div_sin' stops working (returns NaNs because one is dividing by ZERO)!
    """

    # define enumeration attributes used for Chebyshev-Gauss quadrature test
    ARRAY_VALUES_LARGE = np.ones((10000,), dtype=np.float64).tobytes()
    ARRAY_VALUES = np.ones((500,), dtype=np.float64).tobytes()
    ARRAY_VALUES_MEDIUM = np.ones((200,), dtype=np.float64).tobytes()
    ARRAY_VALUES_SMALL = np.ones((100,), dtype=np.float64).tobytes()
    MU_VALUES_LARGE = np.linspace(
        -0.99999999, 0.99999999, num=10000, dtype=np.float64
    ).tobytes()
    MU_VALUES = np.linspace(
        -0.99999999, 0.99999999, num=500, dtype=np.float64
    ).tobytes()
    MU_VALUES_MEDIUM = np.linspace(
        -0.99999999, 0.99999999, num=200, dtype=np.float64
    ).tobytes()
    MU_VALUES_SMALL = np.linspace(
        -0.99999999, 0.99999999, num=100, dtype=np.float64
    ).tobytes()
    MU_VALUES_WITH_BOUNDS_LARGE = np.linspace(
        -1.0, 1.0, num=10000, dtype=np.float64
    ).tobytes()
    MU_VALUES_WITH_BOUNDS = np.linspace(
        -1.0, 1.0, num=500, dtype=np.float64
    ).tobytes()
    MU_VALUES_WITH_BOUNDS_MEDIUM = np.linspace(
        -1.0, 1.0, num=200, dtype=np.float64
    ).tobytes()
    MU_VALUES_WITH_BOUNDS_SMALL = np.linspace(
        -1.0, 1.0, num=100, dtype=np.float64
    ).tobytes()


# generate the fixture used to set up the test class that
# tests the Chebyshev Gauss quadrature
@pytest.fixture(scope='class', params=[5, 10, 25, 50, 100, 500, 1000, 5000])
def setup_test_class_cheby_gauss(request) -> None:
    """Generates a fixture that stores test class variables."""
    # store the expected integral result
    request.cls.result: float = np.pi
    # store the parameter = Chebyshev-Gauss quadrature order
    request.cls.order: int = request.param
    # get the enumeration data dictionary
    request.cls.data_dict: dict[
        str, np.ndarray
    ] = InputDataExampleChebyshevGauss.get_enumeration_dict()
    # store the expected integral result for x^2 as the integrand
    request.cls.result_sq: float = np.pi / 2.0
    # store the expected integral result for x^3 as the integrand
    request.cls.result_cube: float = 0.0
    # store the expected integral result for x^2*(1-x^2) as the integrand
    request.cls.result_no_singular: float = np.pi / 8.0


@pytest.mark.usefixtures('setup_test_class_cheby_gauss')
class TestChebyshevGaussQuadrature:
    """Tests the Chebyshev-Gauss quadrature method."""

    # attribute type declarations
    result: float
    order: int
    data_dict: dict[str, np.ndarray]
    result_sq: float
    result_cube: float
    result_no_singular: float

    def test_without_bounds(self):
        """Tests the Chebyshev-Gauss quadrature method without inclusion of bounds in the interpolation interval."""
        # -- f(x) = 1
        # generate the interpolation function
        _my_interp = interp1d(
            self.data_dict['mu_values'], self.data_dict['array_values']
        )
        # use the interpolation function to get the
        # chebyshev-gauss quadrature result
        cg_result = HI._chebyshev_gauss_quad(_my_interp, order=self.order)
        # assert the result is OK
        assert math.isclose(cg_result, self.result)
        # -- f(x) = x^2
        # generate the interpolation function
        _my_interp_sq = interp1d(
            self.data_dict['mu_values'], self.data_dict['mu_values'] ** 2.0
        )
        # use the interpolation function to get the
        # chebyshev-gauss quadrature result
        cg_result_sq = HI._chebyshev_gauss_quad(_my_interp_sq, order=self.order)
        # assert the result is OK
        assert math.isclose(cg_result_sq, self.result_sq, rel_tol=1e-5)
        # -- f(x) = x^3
        # generate the interpolation function
        _my_interp_cube = interp1d(
            self.data_dict['mu_values'], self.data_dict['mu_values'] ** 3.0
        )
        # use the interpolation function to get the
        # chebyshev-gauss quadrature result
        cg_result_cube = HI._chebyshev_gauss_quad(
            _my_interp_cube, order=self.order
        )
        # assert the result is OK
        assert math.isclose(cg_result_cube, self.result_cube, abs_tol=1e-15)
        # -- f(x) = x^2*(1-x^2)
        # generate the interpolation function
        _my_interp_no_singular = interp1d(
            self.data_dict['mu_values'],
            (self.data_dict['mu_values'] ** 2.0)
            * (1.0 - (self.data_dict['mu_values'] ** 2.0)),
        )
        # use the interpolation function to get the
        # chebyshev-gauss quadrature result
        cg_result_no_singular = HI._chebyshev_gauss_quad(
            _my_interp_no_singular, order=self.order
        )
        # assert the result is OK
        assert math.isclose(
            cg_result_no_singular, self.result_no_singular, rel_tol=1e-4
        )

    def test_with_bounds(self):
        """Tests the Chebyshev-Gauss quadrature method with inclusion of bounds in the interpolation interval."""
        # -- f(x) = 1
        # generate the interpolation function
        _my_interp = interp1d(
            self.data_dict['mu_values_with_bounds'],
            self.data_dict['array_values'],
        )
        # use the interpolation function to get the
        # chebyshev-gauss quadrature result
        cg_result = HI._chebyshev_gauss_quad(_my_interp, order=self.order)
        # assert the result is OK
        assert math.isclose(cg_result, self.result)
        # -- f(x) = x^2
        # generate the interpolation function
        _my_interp_sq = interp1d(
            self.data_dict['mu_values_with_bounds'],
            self.data_dict['mu_values_with_bounds'] ** 2.0,
        )
        # use the interpolation function to get the
        # chebyshev-gauss quadrature result
        cg_result_sq = HI._chebyshev_gauss_quad(_my_interp_sq, order=self.order)
        # assert the result is OK
        assert math.isclose(cg_result_sq, self.result_sq, rel_tol=1e-5)
        # -- f(x) = x^3
        # generate the interpolation function
        _my_interp_cube = interp1d(
            self.data_dict['mu_values_with_bounds'],
            self.data_dict['mu_values_with_bounds'] ** 3.0,
        )
        # use the interpolation function to get the
        # chebyshev-gauss quadrature result
        cg_result_cube = HI._chebyshev_gauss_quad(
            _my_interp_cube, order=self.order
        )
        # assert the result is OK
        assert math.isclose(cg_result_cube, self.result_cube, abs_tol=1e-15)
        # -- f(x) = x^2*(1-x^2)
        # generate the interpolation function
        _my_interp_no_singular = interp1d(
            self.data_dict['mu_values_with_bounds'],
            (self.data_dict['mu_values_with_bounds'] ** 2.0)
            * (1.0 - (self.data_dict['mu_values_with_bounds'] ** 2.0)),
        )
        # use the interpolation function to get the
        # chebyshev-gauss quadrature result
        cg_result_no_singular = HI._chebyshev_gauss_quad(
            _my_interp_no_singular, order=self.order
        )
        # assert the result is OK
        assert math.isclose(
            cg_result_no_singular, self.result_no_singular, rel_tol=1e-4
        )

    def test_without_bounds_large(self):
        """Tests the Chebyshev-Gauss quadrature method without inclusion of bounds in the interpolation interval with a large data array."""
        # -- f(x) = 1
        # generate the interpolation function
        _my_interp = interp1d(
            self.data_dict['mu_values_large'],
            self.data_dict['array_values_large'],
        )
        # use the interpolation function to get the
        # chebyshev-gauss quadrature result
        cg_result = HI._chebyshev_gauss_quad(_my_interp, order=self.order)
        # assert the result is OK
        assert math.isclose(cg_result, self.result)
        # -- f(x) = x^2
        # generate the interpolation function
        _my_interp_sq = interp1d(
            self.data_dict['mu_values_large'],
            self.data_dict['mu_values_large'] ** 2.0,
        )
        # use the interpolation function to get the
        # chebyshev-gauss quadrature result
        cg_result_sq = HI._chebyshev_gauss_quad(_my_interp_sq, order=self.order)
        # assert the result is OK
        assert math.isclose(cg_result_sq, self.result_sq, rel_tol=1e-7)
        # -- f(x) = x^3
        # generate the interpolation function
        _my_interp_cube = interp1d(
            self.data_dict['mu_values_large'],
            self.data_dict['mu_values_large'] ** 3.0,
        )
        # use the interpolation function to get the
        # chebyshev-gauss quadrature result
        cg_result_cube = HI._chebyshev_gauss_quad(
            _my_interp_cube, order=self.order
        )
        # assert the result is OK
        assert math.isclose(cg_result_cube, self.result_cube, abs_tol=1e-15)
        # -- f(x) = x^2*(1-x^2)
        # generate the interpolation function
        _my_interp_no_singular = interp1d(
            self.data_dict['mu_values_large'],
            (self.data_dict['mu_values_large'] ** 2.0)
            * (1.0 - (self.data_dict['mu_values_large'] ** 2.0)),
        )
        # use the interpolation function to get the
        # chebyshev-gauss quadrature result
        cg_result_no_singular = HI._chebyshev_gauss_quad(
            _my_interp_no_singular, order=self.order
        )
        # assert the result is OK
        assert math.isclose(
            cg_result_no_singular, self.result_no_singular, rel_tol=1e-6
        )

    def test_with_bounds_large(self):
        """Tests the Chebyshev-Gauss quadrature method with inclusion of bounds in the interpolation interval with a large data array."""
        # -- f(x) = 1
        # generate the interpolation function
        _my_interp = interp1d(
            self.data_dict['mu_values_with_bounds_large'],
            self.data_dict['array_values_large'],
        )
        # use the interpolation function to get the
        # chebyshev-gauss quadrature result
        cg_result = HI._chebyshev_gauss_quad(_my_interp, order=self.order)
        # assert the result is OK
        assert math.isclose(cg_result, self.result)
        # -- f(x) = x^2
        # generate the interpolation function
        _my_interp_sq = interp1d(
            self.data_dict['mu_values_with_bounds_large'],
            self.data_dict['mu_values_with_bounds_large'] ** 2.0,
        )
        # use the interpolation function to get the
        # chebyshev-gauss quadrature result
        cg_result_sq = HI._chebyshev_gauss_quad(_my_interp_sq, order=self.order)
        # assert the result is OK
        assert math.isclose(cg_result_sq, self.result_sq, rel_tol=1e-7)
        # -- f(x) = x^3
        # generate the interpolation function
        _my_interp_cube = interp1d(
            self.data_dict['mu_values_with_bounds_large'],
            self.data_dict['mu_values_with_bounds_large'] ** 3.0,
        )
        # use the interpolation function to get the
        # chebyshev-gauss quadrature result
        cg_result_cube = HI._chebyshev_gauss_quad(
            _my_interp_cube, order=self.order
        )
        # assert the result is OK
        assert math.isclose(cg_result_cube, self.result_cube, abs_tol=1e-15)
        # -- f(x) = x^2*(1-x^2)
        # generate the interpolation function
        _my_interp_no_singular = interp1d(
            self.data_dict['mu_values_with_bounds_large'],
            (self.data_dict['mu_values_with_bounds_large'] ** 2.0)
            * (1.0 - (self.data_dict['mu_values_with_bounds_large'] ** 2.0)),
        )
        # use the interpolation function to get the
        # chebyshev-gauss quadrature result
        cg_result_no_singular = HI._chebyshev_gauss_quad(
            _my_interp_no_singular, order=self.order
        )
        # assert the result is OK
        assert math.isclose(
            cg_result_no_singular, self.result_no_singular, rel_tol=1e-6
        )

    def test_without_bounds_medium(self):
        """Tests the Chebyshev-Gauss quadrature method with inclusion of bounds in the interpolation interval using a smaller data array."""
        # -- f(x) = 1
        # generate the interpolation function
        _my_interp = interp1d(
            self.data_dict['mu_values_medium'],
            self.data_dict['array_values_medium'],
        )
        # use the interpolation function to get the
        # chebyshev-gauss quadrature result
        cg_result = HI._chebyshev_gauss_quad(_my_interp, order=self.order)
        # assert the result is OK
        assert math.isclose(cg_result, self.result)
        # -- f(x) = x^2
        # generate the interpolation function
        _my_interp_sq = interp1d(
            self.data_dict['mu_values_medium'],
            self.data_dict['mu_values_medium'] ** 2.0,
        )
        # use the interpolation function to get the
        # chebyshev-gauss quadrature result
        cg_result_sq = HI._chebyshev_gauss_quad(_my_interp_sq, order=self.order)
        # assert the result is OK
        assert math.isclose(cg_result_sq, self.result_sq, rel_tol=1e-4)
        # -- f(x) = x^3
        # generate the interpolation function
        _my_interp_cube = interp1d(
            self.data_dict['mu_values_medium'],
            self.data_dict['mu_values_medium'] ** 3.0,
        )
        # use the interpolation function to get the
        # chebyshev-gauss quadrature result
        cg_result_cube = HI._chebyshev_gauss_quad(
            _my_interp_cube, order=self.order
        )
        # assert the result is OK
        assert math.isclose(cg_result_cube, self.result_cube, abs_tol=1e-15)
        # -- f(x) = x^2*(1-x^2)
        # generate the interpolation function
        _my_interp_no_singular = interp1d(
            self.data_dict['mu_values_medium'],
            (self.data_dict['mu_values_medium'] ** 2.0)
            * (1.0 - (self.data_dict['mu_values_medium'] ** 2.0)),
        )
        # use the interpolation function to get the
        # chebyshev-gauss quadrature result
        cg_result_no_singular = HI._chebyshev_gauss_quad(
            _my_interp_no_singular, order=self.order
        )
        # assert the result is OK
        assert math.isclose(
            cg_result_no_singular, self.result_no_singular, rel_tol=1e-3
        )

    def test_with_bounds_medium(self):
        """Tests the Chebyshev-Gauss quadrature method with inclusion of bounds in the interpolation interval using a smaller data array."""
        # -- f(x) = 1
        # generate the interpolation function
        _my_interp = interp1d(
            self.data_dict['mu_values_with_bounds_medium'],
            self.data_dict['array_values_medium'],
        )
        # use the interpolation function to get the
        # chebyshev-gauss quadrature result
        cg_result = HI._chebyshev_gauss_quad(_my_interp, order=self.order)
        # assert the result is OK
        assert math.isclose(cg_result, self.result)
        # -- f(x) = x^2
        # generate the interpolation function
        _my_interp_sq = interp1d(
            self.data_dict['mu_values_with_bounds_medium'],
            self.data_dict['mu_values_with_bounds_medium'] ** 2.0,
        )
        # use the interpolation function to get the
        # chebyshev-gauss quadrature result
        cg_result_sq = HI._chebyshev_gauss_quad(_my_interp_sq, order=self.order)
        # assert the result is OK
        assert math.isclose(cg_result_sq, self.result_sq, rel_tol=1e-4)
        # -- f(x) = x^3
        # generate the interpolation function
        _my_interp_cube = interp1d(
            self.data_dict['mu_values_with_bounds_medium'],
            self.data_dict['mu_values_with_bounds_medium'] ** 3.0,
        )
        # use the interpolation function to get the
        # chebyshev-gauss quadrature result
        cg_result_cube = HI._chebyshev_gauss_quad(
            _my_interp_cube, order=self.order
        )
        # assert the result is OK
        assert math.isclose(cg_result_cube, self.result_cube, abs_tol=1e-15)
        # -- f(x) = x^2*(1-x^2)
        # generate the interpolation function
        _my_interp_no_singular = interp1d(
            self.data_dict['mu_values_with_bounds_medium'],
            (self.data_dict['mu_values_with_bounds_medium'] ** 2.0)
            * (1.0 - (self.data_dict['mu_values_with_bounds_medium'] ** 2.0)),
        )
        # use the interpolation function to get the
        # chebyshev-gauss quadrature result
        cg_result_no_singular = HI._chebyshev_gauss_quad(
            _my_interp_no_singular, order=self.order
        )
        # assert the result is OK
        assert math.isclose(
            cg_result_no_singular, self.result_no_singular, rel_tol=1e-3
        )

    def test_without_bounds_small(self):
        """Tests the Chebyshev-Gauss quadrature method with inclusion of bounds in the interpolation interval using a smaller data array."""
        # -- f(x) = 1
        # generate the interpolation function
        _my_interp = interp1d(
            self.data_dict['mu_values_small'],
            self.data_dict['array_values_small'],
        )
        # use the interpolation function to get the
        # chebyshev-gauss quadrature result
        cg_result = HI._chebyshev_gauss_quad(_my_interp, order=self.order)
        # assert the result is OK
        assert math.isclose(cg_result, self.result)
        # -- f(x) = x^2
        # generate the interpolation function
        _my_interp_sq = interp1d(
            self.data_dict['mu_values_small'],
            self.data_dict['mu_values_small'] ** 2.0,
        )
        # use the interpolation function to get the
        # chebyshev-gauss quadrature result
        cg_result_sq = HI._chebyshev_gauss_quad(_my_interp_sq, order=self.order)
        # assert the result is OK
        assert math.isclose(cg_result_sq, self.result_sq, rel_tol=1e-3)
        # -- f(x) = x^3
        # generate the interpolation function
        _my_interp_cube = interp1d(
            self.data_dict['mu_values_small'],
            self.data_dict['mu_values_small'] ** 3.0,
        )
        # use the interpolation function to get the
        # chebyshev-gauss quadrature result
        cg_result_cube = HI._chebyshev_gauss_quad(
            _my_interp_cube, order=self.order
        )
        # assert the result is OK
        assert math.isclose(cg_result_cube, self.result_cube, abs_tol=1e-15)
        # -- f(x) = x^2*(1-x^2)
        # generate the interpolation function
        _my_interp_no_singular = interp1d(
            self.data_dict['mu_values_small'],
            (self.data_dict['mu_values_small'] ** 2.0)
            * (1.0 - (self.data_dict['mu_values_small'] ** 2.0)),
        )
        # use the interpolation function to get the
        # chebyshev-gauss quadrature result
        cg_result_no_singular = HI._chebyshev_gauss_quad(
            _my_interp_no_singular, order=self.order
        )
        # assert the result is OK
        assert math.isclose(
            cg_result_no_singular, self.result_no_singular, rel_tol=1e-2
        )

    def test_with_bounds_small(self):
        """Tests the Chebyshev-Gauss quadrature method with inclusion of bounds in the interpolation interval using a smaller data array."""
        # -- f(x) = 1
        # generate the interpolation function
        _my_interp = interp1d(
            self.data_dict['mu_values_with_bounds_small'],
            self.data_dict['array_values_small'],
        )
        # use the interpolation function to get the
        # chebyshev-gauss quadrature result
        cg_result = HI._chebyshev_gauss_quad(_my_interp, order=self.order)
        # assert the result is OK
        assert math.isclose(cg_result, self.result)
        # -- f(x) = x^2
        # generate the interpolation function
        _my_interp_sq = interp1d(
            self.data_dict['mu_values_with_bounds_small'],
            self.data_dict['mu_values_with_bounds_small'] ** 2.0,
        )
        # use the interpolation function to get the
        # chebyshev-gauss quadrature result
        cg_result_sq = HI._chebyshev_gauss_quad(_my_interp_sq, order=self.order)
        # assert the result is OK
        assert math.isclose(cg_result_sq, self.result_sq, rel_tol=1e-3)
        # -- f(x) = x^3
        # generate the interpolation function
        _my_interp_cube = interp1d(
            self.data_dict['mu_values_with_bounds_small'],
            self.data_dict['mu_values_with_bounds_small'] ** 3.0,
        )
        # use the interpolation function to get the
        # chebyshev-gauss quadrature result
        cg_result_cube = HI._chebyshev_gauss_quad(
            _my_interp_cube, order=self.order
        )
        # assert the result is OK
        assert math.isclose(cg_result_cube, self.result_cube, abs_tol=1e-15)
        # -- f(x) = x^2*(1-x^2)
        # generate the interpolation function
        _my_interp_no_singular = interp1d(
            self.data_dict['mu_values_with_bounds_small'],
            (self.data_dict['mu_values_with_bounds_small'] ** 2.0)
            * (1.0 - (self.data_dict['mu_values_with_bounds_small'] ** 2.0)),
        )
        # use the interpolation function to get the
        # chebyshev-gauss quadrature result
        cg_result_no_singular = HI._chebyshev_gauss_quad(
            _my_interp_no_singular, order=self.order
        )
        # assert the result is OK
        assert math.isclose(
            cg_result_no_singular, self.result_no_singular, rel_tol=1e-2
        )
