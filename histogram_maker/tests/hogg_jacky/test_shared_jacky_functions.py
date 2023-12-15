"""Python test module for the shared jacky functions.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import modules
import math
import pytest
import numpy as np

# import the functions to be tested
from hogg_jacky.shared import (
    print_single_value,
    print_maximal_values,
    retrieve_valued_data,
    compute_bin_width,
    get_parameter_combinations,
    get_peak_to_peak_range,
    handle_single_data_point,
    check_data_shape,
    check_input_data,
)


def test_print_single_valid_value(capfd) -> None:
    """Captures print statements made by the 'print_single_value' functions and checks their validity."""
    # print a single value
    print_single_value(custom_bin_width=1.0, ones_combo=(2.0, 1))
    # read the stdout
    out, _ = capfd.readouterr()
    # assert it is correct
    assert (
        out
        == 'The likelihood value was not computed because there is only 1 data point available! We used a custom bin width with a value of 1.0, a default alpha parameter with a value of 2.0 and a single bin (number of bins = 1).\n'
    )


def test_print_single_invalid_value(capfd) -> None:
    """Captures print statements made by the 'print_single_value' functions and checks their validity."""
    # print a single value
    print_single_value(custom_bin_width=1.0, ones_combo=None)
    # read the stdout
    out, _ = capfd.readouterr()
    # assert it is correct
    assert (
        out
        == 'The likelihood value was not computed because there is only 1 data point available! We used a custom bin width with a value of 1.0, a default alpha parameter with a value of None and a single bin (number of bins = None).\n'
    )


def test_print_maximal_values(capfd) -> None:
    """Captures print statements made by the 'print_maximal_values' functions and checks their validity."""
    # print a single value
    print_maximal_values(max_bin_width_edges=1.0, max_combo=(2.0, 5))
    # read the stdout
    out, _ = capfd.readouterr()
    print(out)
    # assert it is correct
    assert (
        out
        == 'The likelihood value was maximized for a bin width of 1.0, an alpha parameter with a value of 2.0 and 5 bins.\n'
    )


@pytest.mark.parametrize(
    'nr_nan_values,expected_zeros', [(15, 35), (45, 5), (50, 0)]
)
def test_retrieve_valued_data(nr_nan_values: int, expected_zeros: int) -> None:
    """Tests the 'retrieve_valued_data' function.

    Parameters
    ----------
    nr_nan_values : int
        The number of NaN values to be inserted in a numpy array containing 50 zeroes.
    expected_zeros : int
        The number of expected zeroes in the masked output array.
    """
    # create numpy array containing 50 zeros
    test_arr = np.zeros((50,), dtype=np.float64)
    # add 'nr_nan_values' NaN values at random positions
    test_arr[
        np.random.choice(test_arr.size, nr_nan_values, replace=False)
    ] = np.nan
    # compute the nan mask
    my_nan_mask = np.isnan(test_arr)
    # use the function
    function_result = retrieve_valued_data(
        my_data=test_arr, nan_mask=my_nan_mask
    )
    # check that the valued data contains 50 - 'nr_nan_values' zeroes
    assert np.allclose(
        function_result, np.zeros((expected_zeros,), dtype=np.float64)
    )


def test_compute_bin_width() -> None:
    """Tests the 'compute_bin_width' function."""
    # define the input value range and nr. of bins
    input_value_range = 2.0
    nr_bins = 8
    # compute the bin width using the function
    computed_bin_width = compute_bin_width(
        my_ptp_range=input_value_range, my_bin_number=nr_bins
    )
    # check that the computed bin width is OK
    assert math.isclose(computed_bin_width, 0.25)


@pytest.mark.parametrize('return_list_value', [True, False])
def test_get_parameter_combinations(return_list_value: bool) -> None:
    """Tests the 'get_parameter_combinations' function.

    Parameters
    ----------
    return_list_value : bool
        Defines if the function returns a list or an iterator.
    """
    # define the input alpha parameters, nrs. of bins
    alpha_vals = np.array([0.1, 1.0, 10.0], dtype=np.float64)
    bin_numbers = np.array([3, 4, 5], dtype=np.int32)
    # get the cartesian product combinations of the input values
    cartesian_product_combinations = get_parameter_combinations(
        my_alpha_parameters=alpha_vals,
        my_bin_numbers=bin_numbers,
        return_list=return_list_value,
    )
    # define expected output
    expected_output = [
        (0.1, 3),
        (0.1, 4),
        (0.1, 5),
        (1.0, 3),
        (1.0, 4),
        (1.0, 5),
        (10.0, 3),
        (10.0, 4),
        (10.0, 5),
    ]
    # assert that the output list is OK
    for _i, (_a, _bn) in enumerate(cartesian_product_combinations):
        assert math.isclose(_a, expected_output[_i][0])
        assert _bn == expected_output[_i][1]


def test_get_ptp_range() -> None:
    """Tests the 'get_peak_to_peak_range' function."""
    # construct a test array
    test_arr = np.linspace(-1.0, 5.0, endpoint=True)
    # retrieve the ptp range
    my_data_range = get_peak_to_peak_range(my_data=test_arr)
    # assert data range is OK
    assert math.isclose(my_data_range, 6.0)


@pytest.mark.parametrize(
    'test_array_input,expected_tuple',
    [
        ([1.0], (1.0, (1.0, 1))),
        ([1.0, 1.0, 1.0], (None, None)),
        ([], (None, None)),
    ],
)
def test_handle_single_data_point(
    test_array_input: list[float] | list,
    expected_tuple: tuple[float, int] | tuple[None, None],
) -> None:
    """Parametrized test of the 'handle_single_data_point' function for a 1D numpy array.

    Parameters
    ----------
    test_array_input : list[float] | list
        Contains the input values for the 1D numpy array.
    expected_tuple : tuple[float, int] | tuple[None, None]
        Contains the expected alpha value (float) and nr. of bins (int) for a 1D numpy array with a single entry. If that 1D numpy array has no entries or more than one entry, a tuple of None values is expected to be returned.
    """
    # create single-valued array
    test_arr = np.array(test_array_input, dtype=np.float64)
    # define the custom bin width used in this case
    my_custom_bin_width = [1.0]
    # test the function
    my_bw, my_combo_tuple = handle_single_data_point(
        my_data=test_arr, custom_bin_width=my_custom_bin_width
    )
    # assert the output is OK
    try:
        assert math.isclose(my_bw, expected_tuple[0])
        assert math.isclose(my_combo_tuple[0], expected_tuple[1][0])
        assert my_combo_tuple[1] == expected_tuple[1][1]
    except TypeError:
        assert my_bw is None
        assert my_combo_tuple is None


@pytest.mark.parametrize(
    'my_data,expected_result',
    [(np.zeros((50,)), True), (np.zeros((0,)), False)],
)
def test_check_data_shape(my_data: np.ndarray, expected_result: bool) -> None:
    """Tests the 'check_data_shape' function.

    Parameters
    ----------
    my_data : np.ndarray
        Input data array.
    expected_result : bool
        Expected result for the function.
    """
    # run the function
    my_output = check_data_shape(my_data=my_data)
    # assert the output is OK
    assert my_output == expected_result


@pytest.mark.parametrize(
    'my_data,my_alphas,my_bin_numbers',
    [
        (np.zeros((50,)), np.zeros((50,)), np.zeros((50,), dtype=np.int32)),
        (np.zeros((50,)), np.zeros((50,)), None),
        (np.zeros((50,)), None, np.zeros((50,), dtype=np.int32)),
        (None, np.zeros((50,)), np.zeros((50,), dtype=np.int32)),
    ],
)
def test_check_input_data(
    caplog,
    my_data: None | np.ndarray,
    my_alphas: None | np.ndarray,
    my_bin_numbers: np.ndarray | None,
) -> None:
    """Tests the function 'check_input_data'.

    Parameters
    ----------
    my_data : None | np.ndarray
        Input data array.
    my_alphas : None | np.ndarray
        Input data alphas.
    my_bin_numbers : np.ndarray | None
        Input data bin numbers.
    expected_result : bool
        Expected result for the function.
    """
    # use the input run the function
    my_output_bool = check_input_data(
        my_data=my_data,
        my_alpha_parameters=my_alphas,
        my_bin_numbers=my_bin_numbers,
    )
    # assert that the result is OK
    if len(caplog.records) > 0:
        # output printed to stdout or stderr, hence, the boolean should be False
        assert my_output_bool is False
    else:
        # no output printed to stdout or stderr, boolean should be True
        assert my_output_bool is True
