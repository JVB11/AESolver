"""Pytest module containing tests for the different jackknife-likelihood-computing implementations.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import modules
import pytest
import numpy as np

# import functionalities to be tested
from hogg_jacky import (
    histogram_likelihood_blas,
    log_histogram_likelihood_blas,
    histogram_likelihood_iter,
    histogram_likelihood_numexpr,
    histogram_likelihood_numpy,
    log_histogram_likelihood_iter,
    log_histogram_likelihood_numexpr,
)


@pytest.fixture(scope='session')
def alpha_vals() -> np.ndarray:
    return np.array([0.1, 1.0, 10.0], dtype=np.float64)


@pytest.fixture(scope='session')
def first_array() -> np.ndarray:
    return np.array(
        [1.0, 2.2, 3.0, 4.4, 5.0, 6.3, 7.0, 8.1, 9.0, 10.5], dtype=np.float64
    )


@pytest.fixture(scope='session')
def second_array() -> np.ndarray:
    return np.array(
        [1.0, 2.2, 3.0, 4.4, 5.0, 6.3, 7.0, 8.1, 9.0, 10.5, 11.8],
        dtype=np.float64,
    )


def test_blas(
    alpha_vals: np.ndarray, first_array: np.ndarray, second_array: np.ndarray
) -> None:
    """Test the BLAS implementation."""
    my_bin_nrs = np.array(range(1, first_array.shape[0] + 1), dtype=np.int32)
    hist, _, _, _ = histogram_likelihood_blas(
        my_data=first_array,
        my_alpha_parameters=alpha_vals,
        my_bin_numbers=my_bin_nrs,
    )
    assert hist.shape[0] == 1
    my_bin_nrs_second = np.array(
        range(2, second_array.shape[0] + 1), dtype=np.int32
    )
    hist_second, _, _, _ = histogram_likelihood_blas(
        my_data=second_array,
        my_alpha_parameters=alpha_vals,
        my_bin_numbers=my_bin_nrs_second,
    )
    assert hist_second.shape[0] == 2


def test_log_blas(
    alpha_vals: np.ndarray, first_array: np.ndarray, second_array: np.ndarray
) -> None:
    """Test the logarithmic BLAS implementation."""
    my_bin_nrs = np.array(range(1, first_array.shape[0] + 1), dtype=np.int32)
    hist, _, _, _ = log_histogram_likelihood_blas(
        my_data=first_array,
        my_alpha_parameters=alpha_vals,
        my_bin_numbers=my_bin_nrs,
    )
    assert hist.shape[0] == 1
    my_bin_nrs_second = np.array(
        range(2, second_array.shape[0] + 1), dtype=np.int32
    )
    hist_second, _, _, _ = log_histogram_likelihood_blas(
        my_data=second_array,
        my_alpha_parameters=alpha_vals,
        my_bin_numbers=my_bin_nrs_second,
    )
    assert hist_second.shape[0] == 2


def test_iter(
    alpha_vals: np.ndarray, first_array: np.ndarray, second_array: np.ndarray
) -> None:
    """Test the iter implementation."""
    my_bin_nrs = np.array(range(1, first_array.shape[0] + 1), dtype=np.int32)
    hist, _, _, _ = histogram_likelihood_iter(
        my_data=first_array,
        my_alpha_parameters=alpha_vals,
        my_bin_numbers=my_bin_nrs,
    )
    assert hist.shape[0] == 1
    my_bin_nrs_second = np.array(
        range(2, second_array.shape[0] + 1), dtype=np.int32
    )
    hist_second, _, _, _ = histogram_likelihood_iter(
        my_data=second_array,
        my_alpha_parameters=alpha_vals,
        my_bin_numbers=my_bin_nrs_second,
    )
    assert hist_second.shape[0] == 2


def test_log_iter(
    alpha_vals: np.ndarray, first_array: np.ndarray, second_array: np.ndarray
) -> None:
    """Test the iter implementation."""
    my_bin_nrs = np.array(range(1, first_array.shape[0] + 1), dtype=np.int32)
    hist, _, _, _ = log_histogram_likelihood_iter(
        my_data=first_array,
        my_alpha_parameters=alpha_vals,
        my_bin_numbers=my_bin_nrs,
    )
    assert hist.shape[0] == 1
    my_bin_nrs_second = np.array(
        range(2, second_array.shape[0] + 1), dtype=np.int32
    )
    hist_second, _, _, _ = log_histogram_likelihood_iter(
        my_data=second_array,
        my_alpha_parameters=alpha_vals,
        my_bin_numbers=my_bin_nrs_second,
    )
    assert hist_second.shape[0] == 2


def test_numexpr(
    alpha_vals: np.ndarray, first_array: np.ndarray, second_array: np.ndarray
) -> None:
    """Test the numexpr implementation."""
    my_bin_nrs = np.array(range(1, first_array.shape[0] + 1), dtype=np.int32)
    hist, _, _, _ = histogram_likelihood_numexpr(
        my_data=first_array,
        my_alpha_parameters=alpha_vals,
        my_bin_numbers=my_bin_nrs,
    )
    assert hist.shape[0] == 1
    my_bin_nrs_second = np.array(
        range(2, second_array.shape[0] + 1), dtype=np.int32
    )
    hist_second, _, _, _ = histogram_likelihood_numexpr(
        my_data=second_array,
        my_alpha_parameters=alpha_vals,
        my_bin_numbers=my_bin_nrs_second,
    )
    assert hist_second.shape[0] == 2


def test_log_numexpr(
    alpha_vals: np.ndarray, first_array: np.ndarray, second_array: np.ndarray
) -> None:
    """Test the numexpr implementation."""
    my_bin_nrs = np.array(range(1, first_array.shape[0] + 1), dtype=np.int32)
    hist, _, _, _ = log_histogram_likelihood_numexpr(
        my_data=first_array,
        my_alpha_parameters=alpha_vals,
        my_bin_numbers=my_bin_nrs,
    )
    assert hist.shape[0] == 1
    my_bin_nrs_second = np.array(
        range(2, second_array.shape[0] + 1), dtype=np.int32
    )
    hist_second, _, _, _ = log_histogram_likelihood_numexpr(
        my_data=second_array,
        my_alpha_parameters=alpha_vals,
        my_bin_numbers=my_bin_nrs_second,
    )
    assert hist_second.shape[0] == 2


def test_numpy(
    alpha_vals: np.ndarray, first_array: np.ndarray, second_array: np.ndarray
) -> None:
    """Test the numpy implementation."""
    my_bin_nrs = np.array(range(1, first_array.shape[0] + 1), dtype=np.int32)
    hist, _, _, _ = histogram_likelihood_numpy(
        my_data=first_array,
        my_alpha_parameters=alpha_vals,
        my_bin_numbers=my_bin_nrs,
    )
    assert hist.shape[0] == 1
    my_bin_nrs_second = np.array(
        range(2, second_array.shape[0] + 1), dtype=np.int32
    )
    hist_second, _, _, _ = histogram_likelihood_numpy(
        my_data=second_array,
        my_alpha_parameters=alpha_vals,
        my_bin_numbers=my_bin_nrs_second,
    )
    assert hist_second.shape[0] == 2
