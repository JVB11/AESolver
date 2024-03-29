"""Python script used to compute/interpolate the threshold boundary for theta_2,
based on values of the isolated mode triads listed in Table 5 of Van Beeck et al. (2024).

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import numpy as np
import pandas as pd
from scipy.stats import linregress
import matplotlib.pyplot as plt

# type checking imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scipy.stats import LinregressResult  # type: ignore
    import numpy.typing as npt


def get_values_for_interpolation() -> 'tuple[list[float], list[float]]':
    """Returns values of theta_2 and the daughter-parent amplitude ratio (for daughter 1)
    used to compute/interpolate the threshold boundary.

    Returns
    -------
    theta_2 : list[float]
        Contains values of theta_2.
    ratio_2_1 : list[float]
        Contains values of the daughter-parent amplitude ratio (for daughter 1).
    """
    # store the theta_2 values
    theta_2 = [
        0.76947,
        0.92560,
        0.30289,
        0.17994,
        2.06433,
        1.27865,
        0.01816,
        0.76510,
        0.13080,
        0.00907,
        0.02771,
        0.85748,
        0.09346,
        38.7770,
        3.58916,
        0.21492,
        0.92741,
        0.99670,
        0.41545,
        1.58532,
        18.0122,
    ]
    # store the daughter-parent amplitude ratio (daughter 1)
    ratio_2_1 = [
        0.55605,
        0.42858,
        0.65585,
        0.91831,
        0.48298,
        0.32000,
        2.53183,
        0.32190,
        0.84057,
        3.08804,
        2.02044,
        0.79629,
        2.05217,
        0.07627,
        0.29260,
        0.72069,
        0.34623,
        0.48095,
        0.61223,
        0.30289,
        0.12459,
    ]
    # return the interpolation values
    return theta_2, ratio_2_1


def generate_plot_dataframe(
    theta_2: 'list[float]', ratio_2_1: 'list[float]'
) -> 'pd.DataFrame':
    """Generates the Pandas DataFrame used for plotting the results of the linear regression
    used to estimate the threshold boundary.

    Parameters
    ----------
    theta_2 : list[float]
        Contains values of theta_2.
    ratio_2_1 : list[float]
        Contains values of the daughter-parent amplitude ratio (for daughter 1).

    Returns
    -------
    pd.DataFrame
        DataFrame used for plotting the results of the linear regression used
        to estimate the threshold boundary.
    """
    return pd.DataFrame(
        data=zip(theta_2, ratio_2_1), columns=['theta_2', 'ratio_2_1']
    )


def get_difference_arrays(
    my_df: 'pd.DataFrame'
) -> 'tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]':
    """Generates the difference arrays from the plot DataFrame.

    Parameters
    ----------
    my_df : pd.DataFrame
        DataFrame used for plotting the results of the linear regression used
        to estimate the threshold boundary.

    Returns
    -------
    my_pos_diff : np.ndarray[np.float64]
        Array containing the positive differences of the daughter-parent amplitude ratio with respect to 1.0
        (i.e., those differences computed for daughter-parent amplitude ratios larger than 1.0).
        NaN-valued entries in this array indicate entries in the daughter-parent amplitude ratio array
        that are smaller or equal to 1.0.
    my_neg_diff : np.ndarray[np.float64]
        Array containing the negative differences of the daughter-parent amplitude ratio with respect to 1.0
        (i.e., those differences computed for daughter-parent amplitude ratios smaller than 1.0).
        NaN-valued entries in this array indicate entries in the daughter-parent amplitude ratio array
        that are larger or equal to 1.0.
    """
    # get the necessary difference arrays
    my_diff = my_df.loc[:, 'ratio_2_1'].to_numpy() - 1.0
    # distinguish between positive and negative differences
    pos = my_diff > 0.0
    my_pos_diff = my_diff.copy()
    my_neg_diff = my_diff.copy()
    my_pos_diff[~pos] = np.NaN
    my_neg_diff[pos] = np.NaN
    # return the difference arrays
    return my_pos_diff, my_neg_diff


def get_k_smallest_points_masks(
    k: int,
    my_pos_diff: 'npt.NDArray[np.float64]',
    my_neg_diff: 'npt.NDArray[np.float64]',
) -> 'tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]':
    """Generates the masks used to select the k entries in the daughter-parent amplitude ratio array that are closest
    to 1.0 from both sides (i.e., the k entries closest to 1.0 and smaller than 1.0, as well as
    the k entries closest to 1.0 and larger than 1.0.).

    Parameters
    ----------
    k : int
        Number of daughter-parent amplitude ratio entries considered for the interpolation of the threshold value,
        which are closest to 1.0 (i.e., if k == 2, the 2 entries of the positive differences,
        and the 2 entries of the negative differences closest to 0.0 are considered for the interpolation.).
    my_pos_diff : np.ndarray[np.float64]
        Array containing the positive differences of the daughter-parent amplitude ratio with respect to 1.0
        (i.e., those differences computed for daughter-parent amplitude ratios larger than 1.0).
        NaN-valued entries in this array indicate entries in the daughter-parent amplitude ratio array
        that are smaller or equal to 1.0.
    my_neg_diff : np.ndarray[np.float64]
        Array containing the negative differences of the daughter-parent amplitude ratio with respect to 1.0
        (i.e., those differences computed for daughter-parent amplitude ratios smaller than 1.0).
        NaN-valued entries in this array indicate entries in the daughter-parent amplitude ratio array
        that are larger or equal to 1.0.

    Returns
    -------
    pos_closest_k : np.ndarray[np.int32]
        Integer mask used to select the k entries of the positive differences closest to 0.0.
    neg_closest_k : np.ndarray[np.int32]
        Integer mask used to select the k entries of the negative differences closest to 0.0.
    """
    # get the integer masks
    pos_closest_k = np.argpartition(my_pos_diff, k)
    neg_closest_k = np.argpartition(np.abs(my_neg_diff), k)
    # return them
    return pos_closest_k, neg_closest_k


def get_k_smallest_points(
    k: int,
    X: 'npt.NDArray[np.float64]',
    Y: 'npt.NDArray[np.float64]',
    pos_closest_k: 'npt.NDArray[np.int32]',
    neg_closest_k: 'npt.NDArray[np.int32]',
) -> 'tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]':
    """Retrieves the k entries in the daughter-parent amplitude ratio array that are closest
    to 1.0 from both sides (i.e., the k entries closest to 1.0 and smaller than 1.0,as well as
    the k entries closest to 1.0 and larger than 1.0.).

    Parameters
    ----------
    k : int
        Number of daughter-parent amplitude ratio entries considered for the interpolation of the threshold value,
        which are closest to 1.0 (i.e., if k == 2, the 2 entries of the positive differences,
        and the 2 entries of the negative differences closest to 0.0 are considered for the interpolation.).
    X : np.ndarray[np.float64]
        X-value array used to select the points for the linear regression/interpolation used
        to compute the threshold boundary.
    Y : np.ndarray[np.float64]
        Y-value array used to select the points for the linear regression/interpolation used
        to compute the threshold boundary.
    pos_closest_k : np.ndarray[np.float64]
        Integer mask used to select the k entries of the positive differences closest to 0.0.
    neg_closest_k : np.ndarray[np.float64]
        Integer mask used to select the k entries of the negative differences closest to 0.0.

    Returns
    -------
    xx_k : np.ndarray[np.float64]
        X values used for the linear regression/interpolation with which we compute the threshold boundary.
    yy_k : np.ndarray[np.float64]
        Y values used for the linear regression/interpolation with which we compute the threshold boundary.
    """
    # generate the regression points
    xx_k = np.concatenate([X[pos_closest_k[:k]], X[neg_closest_k[:k]]])
    yy_k = np.concatenate([Y[pos_closest_k[:k]], Y[neg_closest_k[:k]]])
    # return those points
    return xx_k, yy_k


def perform_linear_regression(
    xx_k: 'npt.NDArray[np.float64]', yy_k: 'npt.NDArray[np.float64]'
) -> (
    'tuple[LinregressResult, npt.NDArray[np.float64], npt.NDArray[np.float64]]'
):
    """Performs the linear regression whose results are used to compute the threshold boundary.

    Parameters
    ----------
    xx_k : np.ndarray[np.float64]
        X values used for the linear regression/interpolation with which we compute the threshold boundary.
    yy_k : np.ndarray[np.float64]
        Y values used for the linear regression/interpolation with which we compute the threshold boundary.

    Returns
    -------
    lin_result : LinregressResult
        Result object of the linear regression.
    uncertainty_boundary : np.ndarray[np.float64]
        Uncertainty on the estimated threshold boundary.
    boundary : np.ndarray[np.float64]
        Estimated threshold boundary.
    """
    # perform linear regression
    lin_result = linregress(yy_k, xx_k)
    # compute the linear regression result for the 1.0 boundary
    uncertainty_boundary = np.sqrt(
        lin_result.stderr**2.0 + lin_result.intercept_stderr**2.0  # type: ignore
    )
    boundary = lin_result.intercept + lin_result.slope  # type: ignore
    # return results
    return lin_result, uncertainty_boundary, boundary


def print_result(
    lin_result: 'LinregressResult',
    uncertainty_boundary: 'npt.NDArray[np.float64]',
    boundary: 'npt.NDArray[np.float64]',
) -> 'None':
    """Prints the results of the estimation of the threshold boundary using linear regression.

    Parameters
    ----------
    lin_result : LinregressResult
        Result object of the linear regression.
    uncertainty_boundary : np.ndarray[np.float64]
        Uncertainty on the estimated threshold boundary.
    boundary : np.ndarray[np.float64]
        Estimated threshold boundary.
    """
    print(
        f'\nLinear regression for the threshold boundary yields: {boundary} +/- {uncertainty_boundary}'
    )
    print(f'with regression slope: {lin_result.slope} +/- {lin_result.stderr}')
    print(
        f'and regression intercept: {lin_result.intercept} +/- {lin_result.intercept_stderr}\n'
    )


if __name__ == '__main__':
    # ------------------
    # USER-DEFINED INPUT
    # ------------------
    # define the number of closest points taken into account to
    # determine the boundary (for reproducibility, set to 3)
    my_nr_closest_points = 3
    # ------------------
    # get the theta_2 and daughter-parent amplitude ratios
    theta_2, ratio_2_1 = get_values_for_interpolation()
    # get the relevant dataframe
    my_df = generate_plot_dataframe(theta_2=theta_2, ratio_2_1=ratio_2_1)
    # generate the difference arrays
    my_pos_diff, my_neg_diff = get_difference_arrays(my_df=my_df)
    # get the k smallest point masks on both sides
    pos_closest_k, neg_closest_k = get_k_smallest_points_masks(
        k=my_nr_closest_points, my_neg_diff=my_neg_diff, my_pos_diff=my_pos_diff
    )
    # store the data points (of which a subset will be used
    # for linear regression) in arrays
    X = np.array(theta_2)
    Y = np.array(ratio_2_1)
    # get the k smallest points on both sides
    xx_k, yy_k = get_k_smallest_points(
        k=my_nr_closest_points,
        X=X,
        Y=Y,
        pos_closest_k=pos_closest_k,
        neg_closest_k=neg_closest_k,
    )
    # perform linear regression
    lin_result, uncertainty_boundary, boundary = perform_linear_regression(
        xx_k=xx_k, yy_k=yy_k
    )
    # print the result
    print_result(
        lin_result=lin_result,
        uncertainty_boundary=uncertainty_boundary,
        boundary=boundary,
    )
    # plot the regression result
    plt.plot(lin_result.intercept + lin_result.slope * yy_k, yy_k, 'r')
    # show the result
    plt.show()
