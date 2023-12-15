"""Python test module for the numerical differentiation class used by AESolver.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import pytest
import sys
import numpy as np
from num_deriv import NumericalDifferentiator as Nd  # type: ignore


# generate global linear spaces used for setting up the dictionary
quadratic_linear_space = np.linspace(1, 1000, num=200, dtype=np.float64)
exponential_linear_space = np.linspace(1, 10, num=200, dtype=np.float64)
sine_linear_space = np.linspace(0, 2.0 * np.pi, num=200, dtype=np.float64)


# generate dictionary containing results to be used for pytests
_testing_dictionary_evenly_spaced = {
    'quadratic': {
        'arr_range': quadratic_linear_space,
        'arr_vals': quadratic_linear_space**2.0,  # y = x**2
        'arr_vals_der': 2.0 * quadratic_linear_space,  # y' = 2 * x
        'arr_vals_der_der': 2.0
        * np.ones_like(quadratic_linear_space),  # y'' = 2
    },
    'exponential': {
        'arr_range': exponential_linear_space,
        'arr_vals': np.exp(exponential_linear_space / 10.0),  # y = e^(x/10)
        'arr_vals_der': np.exp(exponential_linear_space / 10.0) / 10.0,
        # y' = e^(x/10) / 10
        'arr_vals_der_der': np.exp(exponential_linear_space / 10.0) / 100.0,
        # y'' = e^(x/10) / 100
    },
    'sine': {
        'arr_range': sine_linear_space,
        'arr_vals': np.sin(sine_linear_space),  # y = sin(x)
        'arr_vals_der': np.cos(sine_linear_space),  # y' = cos(x)
        'arr_vals_der_der': -1.0 * np.sin(sine_linear_space),  # y'' = -sin(x)
    },
}


@pytest.fixture(scope='class', params=['quadratic', 'exponential', 'sine'])
def setup_test_class(request):
    """Generates the fixture for the PyTest test class, based on
    an evenly spaced function and its derivatives.
    """
    # get the correct dictionary
    _my_dict = _testing_dictionary_evenly_spaced[request.param]
    # set the correct class variables
    request.cls.arr_range = _my_dict['arr_range']
    request.cls.arr_vals = _my_dict['arr_vals']
    request.cls.arr_vals_der = _my_dict['arr_vals_der']
    request.cls.arr_vals_der_der = _my_dict['arr_vals_der_der']
    request.cls.fcn_name = request.param


# Generate the PyTest test class, with corresponding fixture
@pytest.mark.usefixtures('setup_test_class')
class TestNumericalDifferentiation:
    """Python class that performs pytests to verify whether
    the Numerical Differentiation class works.
    """

    # attribute type declarations
    arr_range: np.ndarray
    arr_vals: np.ndarray
    arr_vals_der: np.ndarray
    arr_vals_der_der: np.ndarray
    fcn_name: str

    @pytest.mark.parametrize(
        ('diff_method', 'additional_arguments'),
        [
            ('fornberg', {'interpolation_order': 8}),
            pytest.param(
                'gradient',
                {},
                marks=pytest.mark.xfail(
                    reason='Second derivative not supported for the gradient method.\n'
                ),
            ),
            pytest.param(
                'stencil_functions',
                {},
                marks=pytest.mark.xfail(
                    reason='Second derivative not supported for the '
                    'stencil functions method.\n'
                ),
            ),
        ],
    )
    def test_first_and_second_derivative(
        self, diff_method, additional_arguments
    ):
        """Tests first and second derivative for specific differentiation methods.

        Parameters
        ----------
        diff_method : str
            Denotes the differentiation method to be tested.
        additional_arguments : list
            Contains any additional arguments to be passed to the differentiation method.
        """
        # initialize test differentiator object that computes both the
        # first and second derivative in one go
        _differ_second = Nd(
            order_derivative=2, differentiation_method=diff_method
        )
        # numerically differentiate the test values
        _test_diff_vals, _test_diff_text = _differ_second.differentiate(
            x=self.arr_range, y=self.arr_vals, **additional_arguments
        )
        # print info about numerical differentiation test
        print(f'\nNow testing for {self.fcn_name} function:')
        print(_test_diff_text)
        # assert the validity of the first and second numerical derivative
        assert np.allclose(_test_diff_vals[1], self.arr_vals_der)
        assert np.allclose(_test_diff_vals[2], self.arr_vals_der_der)

    @pytest.mark.parametrize(
        ('diff_method', 'additional_arguments', 'cut_ends'),
        [
            ('gradient', {}, False),
            pytest.param(
                'stencil_functions',
                {},
                True,  # TODO: change this when numba supports python3.11 or later!
                marks=pytest.mark.skipif(
                    (sys.version_info[0], sys.version_info[1]) > (3, 10),
                    reason='Numba not supported for py3.11!\n',
                ),
            ),
            pytest.param(
                'stencil_functions',
                {},
                False,
                marks=pytest.mark.xfail(
                    reason='Stencil functions cannot retrieve derivatives at the boundaries'
                    ' of the data array.\n'
                ),
            ),
        ],
    )
    def test_first_derivative_only(
        self, diff_method, additional_arguments, cut_ends
    ):
        """Tests first derivatives for specific differentiation methods whose first derivatives could not yet be tested.

        Parameters
        ----------
        diff_method : str
            Denotes the differentiation method to be tested.
        additional_arguments : list
            Contains any additional arguments to be passed to the differentiation method.
        cut_ends : bool
            If True, cut the ends of the array before derivative computation.
        """
        # initialize test differentiator object that computes the first derivative
        _differ_first = Nd(
            order_derivative=1, differentiation_method=diff_method
        )
        # numerically differentiate the test values
        _test_diff_vals, _test_diff_text = _differ_first.differentiate(
            x=self.arr_range, y=self.arr_vals, **additional_arguments
        )
        # print info about numerical differentiation test
        print(f'\nNow testing for {self.fcn_name} function:')
        print(_test_diff_text)
        # assert the validity of the first numerical derivative
        # - additional xfail when dealing with exponential functions
        if self.fcn_name == 'exponential':
            pytest.xfail(
                'Exponential functions are hard to differentiate. '
                "Use Fornberg's method, which performs ok if numbers "
                'are not too large!\n'
            )
        elif self.fcn_name == 'sine':
            pytest.xfail(
                'Apparently, derivatives of sine functions are '
                'only approximated using these methods...\n'
            )
        # - check if ends need to be cut ('stencil_functions' cannot
        # compute derivative at bounds)!
        if cut_ends:
            print('(Cutting ends to remove boundary effect failures)')
            assert np.allclose(
                _test_diff_vals[1][1:-1], self.arr_vals_der[1:-1]
            )
        else:
            assert np.allclose(_test_diff_vals[1], self.arr_vals_der)
