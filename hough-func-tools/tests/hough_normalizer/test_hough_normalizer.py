"""Pytest module used to test the normalizer object for the Hough function normalization.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import modules
import pytest
import numpy as np
import math

# import pytest utility classes
from pytest_util_classes import EPPs, EVs  # type: ignore

# import the normalizer object to be tested
from hough_function.hough_normalizer import HoughNormalizer


# set up values to be used as input parameters
class InputNormalizations(EPPs):
    """Defines the input normalization formalisms."""

    LEE2012 = None
    PRAT2019 = None
    WRONG = 'Unknown normalization method'


# set up the expected output values for the input parameters
class ExpectedOutput(EVs):
    """Defines the expected output for the test input.
    (Computed using Mathematica)
    """

    LEE2012 = 0.282095
    PRAT2019 = 1.0
    WRONG = None


# set up a fixture for the test class
@pytest.fixture(
    scope='class', params=InputNormalizations.get_all_pytest_parameters()
)
def setup_test_class(request):
    # store the input normalization parameter
    request.cls.norm_strategy = request.param
    # store the arrays to be used for normalization
    request.cls.mu = np.linspace(-1.0, 1.0, dtype=np.float64, num=50)
    request.cls.hr = np.ones(50, dtype=np.float64)
    # store the expected output for this input parameter
    request.cls.expected_output = ExpectedOutput.get_value(request.param)


# define the test class
@pytest.mark.parametrize('omp_threads', [1, 4])
@pytest.mark.parametrize('use_parallel', [False, True])
@pytest.mark.usefixtures('setup_test_class')
class TestHoughNormalizer:
    """Test class for the HoughNormalizer
    object.
    """

    # attribute type declarations
    norm_strategy: str
    mu: np.ndarray
    hr: np.ndarray
    expected_output: float or None

    # test the get_norm_factor method
    def test_get_norm_factor(self, omp_threads, use_parallel):
        """Tests the 'get_norm_factor' method of the HoughNormalizer object.

        Parameters
        ----------
        omp_threads : int
            The nr. of threads to be used in calculations. Only used if use_parallel is True.
        use_parallel : bool
            Whether to use parallelized computations.
        """
        # compute the normalization factors
        my_norm_factor = HoughNormalizer.get_norm_factor(
            norm_procedure=self.norm_strategy,
            mu=self.mu,
            hr=self.hr,
            use_parallel=use_parallel,
            nr_omp_threads=omp_threads,
        )
        # check whether the normalization factor is ok
        if self.expected_output is not None:
            assert math.isclose(
                my_norm_factor,
                self.expected_output,
                abs_tol=1.0e-6,  # OK up to this tolerance!
            )
