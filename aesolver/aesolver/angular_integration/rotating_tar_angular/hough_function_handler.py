"""Python module that contains class that has methods that handle the computation of the Hough functions and their derivatives.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import logging
import numpy as np
from enum import Enum
from operator import attrgetter

# import custom module that computes Hough functions
from hough_function import HF

# type checking imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # import type check types
    from collections.abc import Callable
    from typing import Any

    # numpy typing
    import numpy.typing as npt


# set up the logger
logger = logging.getLogger(__name__)


# ------------------------- HOUGH FUNCTION COMPUTATION -------------------------
# define the class that handles and retrieves Hough functions and their derivatives
class HoughFunctionHandler(Enum):
    """Enumeration class containing methods that handle Hough functions and their derivatives and retrieves them so that they can be stored in the coupling coefficient object."""

    # -- ACCESS DEFINITIONS

    # store the functions used to access Hough functions
    HR: 'Callable[[HF], npt.NDArray[np.float64]]' = attrgetter('hr_range')
    HT: 'Callable[[HF], npt.NDArray[np.float64]]' = attrgetter('ht_range')
    HP: 'Callable[[HF], npt.NDArray[np.float64]]' = attrgetter('hp_range')
    THETA_DER_HR: 'Callable[[HF], npt.NDArray[np.float64]]' = attrgetter(
        'hrp_range'
    )
    THETA_DER2_HR: 'Callable[[HF], npt.NDArray[np.float64]]' = attrgetter(
        'hrpp_range'
    )
    THETA_DER_HT: 'Callable[[HF], npt.NDArray[np.float64]]' = attrgetter(
        'htp_range'
    )
    THETA_DER_HP: 'Callable[[HF], npt.NDArray[np.float64]]' = attrgetter(
        'hpp_range'
    )
    THETA_DER_HR_THETA: 'Callable[[HF], npt.NDArray[np.float64]]' = attrgetter(
        'hrp_range_theta'
    )
    THETA_DER2_HR_THETA: 'Callable[[HF], npt.NDArray[np.float64]]' = attrgetter(
        'hrpp_range_theta'
    )
    THETA_DER_HT_THETA: 'Callable[[HF], npt.NDArray[np.float64]]' = attrgetter(
        'htp_range_theta'
    )
    THETA_DER_HP_THETA: 'Callable[[HF], npt.NDArray[np.float64]]' = attrgetter(
        'hpp_range_theta'
    )
    THETA_PHI: 'Callable[[HF], npt.NDArray[np.float64]]' = attrgetter(
        'theta_phi_hough_tilde'
    )
    PHI_PHI: 'Callable[[HF], npt.NDArray[np.float64]]' = attrgetter(
        'phi_phi_hough_tilde'
    )
    MHR: 'Callable[[HF], npt.NDArray[np.float64]]' = attrgetter('hr_range_m')
    # store the function used to access the mu == cos(theta) array used during Hough function reconstruction
    MU: 'Callable[[HF], npt.NDArray[np.float64]]' = attrgetter('mu_range')
    # store the function used to access the theta array from the mu array used during Hough function reconstruction
    THETAS: 'Callable[[HF], npt.NDArray[np.float64]]' = attrgetter(
        'thetas_range'
    )
    # store the list of names referring to the functions used to access attributes of the Hough functions
    FUNCTION_LIST = [
        'HR',
        'HT',
        'HP',
        'THETA_DER_HR',
        'THETA_DER_HT',
        'THETA_DER_HP',
        'THETA_PHI',
        'PHI_PHI',
        'MHR',
        'THETA_DER2_HR',
        'THETA_DER_HR_THETA',
        'THETA_DER2_HR_THETA',
        'THETA_DER_HT_THETA',
        'THETA_DER_HP_THETA',
    ]

    # class method that retrieves the list of functional names for the Hough functions
    @classmethod
    def get_functional_names(cls) -> list[str]:
        """Class method that retrieves the functional names for the Hough functions.

        Returns
        -------
        list[str]
            The list containing the functional names.
        """
        return [_x.lower() for _x in cls.FUNCTION_LIST.value]

    # class method that computes the Hough function terms + derivatives for the given input
    @classmethod
    def get_relevant_hough_output(
        cls,
        nu: float,
        l: int,  # noqa: E741
        m: int,
        npts: int = 200,
        lmbd: float | None = None,
        use_complex: bool = False,
        lmbd_adjust: bool = True,
        definition: str = 'lee_saio_equivalent',
    ) -> 'tuple[dict, npt.NDArray[np.float64], npt.NDArray[np.float64]]':
        """Class method that computes the requested Hough functions and derivatives for a specific pulsation mode using Chebyshev collocation (see Wang et al. (2016) and Prat et al. (2019)).

        Notes
        -----
        The expression used to compute mode id parameter 'k' is adapted to gravito-inertial modes! When providing estimates of the eigenvalues of the Laplace Tidal Equations (LTE) originating from GYRE, make sure that you set 'lmbd_adjust' to True, because of the difference in definition of LTE for Prat et al. (2019) and Townsend et al. (2020). Do not forget to adjust the 'definition' input to obtain the form of the Hough functions to your liking!

        Parameters
        ----------
        nu : float
            Spin factor for the specific mode.
        l : int
            "Degree" of the mode. This parameter only has meaning in a non-rotating model. Used to compute the appropriate mode id parameter 'k' for gravito-inertial modes in rotating models.
        m : int
            Azimuthal order of the mode.
        npts : int, optional
            Number of points in the collocation grid; by default 200. (similar to Prat et al. 2019).
        lmbd : None or float, optional
            Estimate of the lambda value obtained from e.g. GYRE computations; by default None, which means that the estimate shall be computed as l * (l + 1). Default: None.
        use_complex : bool, optional
            If True, allow complex-valued Hough functions. If False, enforce real-valuedness; by default False.
        lmbd_adjust : bool, optional
            If True, adjust the input lambda/eigenvalue of the Laplace Tidal Equations by multiplying it with a minus sign (see definition of Prat et al. 2019). If False, do not perform such adjustment; by default True.
        definition : str, optional
            Defines which form of the Hough functions you are retrieving; by default 'lee_saio_equivalent'. Options are:

            * 'prat': obtain the form of the Hough functions described in Prat et al. (2019).
            * 'lee_saio_equivalent': obtain the form of the Hough functions described in Lee & Saio (1997), adjusted for the difference in definition of the azimuthal order 'm'.
            * 'townsend': obtain the form of the Hough functions described in Townsend et al. (2020).

        Returns
        -------
        dict
            Contains the necessary Hough functions and derivatives, to be stored in the coupling coefficient object.
        npt.NDArray[np.float64]
            Contains the mu == cos(theta) values used during Hough function reconstruction
        npt.NDArray[np.float64]
            Contains the theta values obtained from the mu values.
        """
        # if a list of lambda values is provided, get the first instance of the list
        if isinstance(lmbd, list) or isinstance(lmbd, np.ndarray):
            lmbd = lmbd[0]
        # get the Hough object for the specific mode
        _my_hough = cls._get_hough_object(
            nu=nu,
            l=l,
            m=m,
            npts=npts,
            lmbd=lmbd,
            use_complex=use_complex,
            lmbd_adjust=lmbd_adjust,
            definition=definition,
        )
        # loop through the different function attributes to retrieve and store the relevant terms for angular integration in a dictionary + return it
        return (
            {
                _hn.lower(): cls._compute_hough_function_or_derivative(
                    action_string=_hn, hough_object=_my_hough
                )
                for _hn in cls.FUNCTION_LIST.value
            },
            cls.MU.value(_my_hough),
            cls.THETAS.value(_my_hough),
        )

    # class method that retrieves the Hough function object for a given input spin factor and mode id
    @classmethod
    def _get_hough_object(
        cls,
        nu: float,
        l: int,  # noqa: E741
        m: int,
        npts: int = 200,
        lmbd: float | None = None,
        use_complex: bool = False,
        lmbd_adjust: bool = True,
        definition: str = 'lee_saio_equivalent',
    ) -> HF:
        """Internal class method that computes the requested Hough function for a specific pulsation mode using Chebyshev collocation (see Wang et al. (2016) and Prat et al. (2019)).

        Notes
        -----
        When providing estimates of the eigenvalues of the Laplace Tidal Equations (LTE) originating from GYRE, make sure that you set 'lmbd_adjust' to True, because of the difference in definition of LTE for Prat et al. (2019) and Townsend et al. (2020). Do not forget to adjust the 'definition' input to obtain the form of the Hough functions to your liking!

        Parameters
        ----------
        nu : float
            Spin factor for the specific mode.
        l : int
            "Degree" of the mode. This parameter only has meaning if non-rotating. Used to compute the appropriate mode id parameter 'k' for gravito-inertial modes.
        m : int
            Azimuthal order of the mode.
        npts : int, optional
            Number of points in the collocation grid; by default 200. (similar to Prat et al. 2019).
        lmbd : None or float, optional
            Estimate of the lambda value obtained from e.g. GYRE computations; by default None, which means that the estimate shall be computed as l * (l + 1).
        use_complex : bool
            If True, allow complex-valued Hough functions. If False, enforce real-valuedness; by default False.
        lmbd_adjust : bool, optional
            If True, adjust the input lambda/eigenvalue of the Laplace Tidal Equations by multiplying it with a minus sign (see definition of Prat et al. 2019). If False, do not perform such adjustment; by default True.
        definition : str, optional
            Defines which form of the Hough functions you are retrieving; by default 'lee_saio_equivalent'. Options are:

            * 'prat': obtain the form of the Hough functions described in Prat et al. (2019).
            * 'lee_saio_equivalent': obtain the form of the Hough functions described in Lee & Saio (1997), adjusted for the difference in definition of the azimuthal order 'm'.
            * 'townsend': obtain the form of the Hough functions described in Townsend et al. (2020).

        Returns
        -------
        HF
            The object containing the requested Hough function data.
        """
        # initialize the object
        hough_object = HF(
            nu=nu,
            l=l,
            m=m,
            npts=npts,
            lmbd=lmbd,
            use_complex=use_complex,
            definition=definition,
            lmbd_adjust=lmbd_adjust,
        )
        # compute the Hough function
        hough_object()
        # return the object
        return hough_object

    # class method that computes a single term of the angular integrand
    @classmethod
    def _compute_hough_function_or_derivative(
        cls, action_string: str, hough_object: HF
    ) -> 'Any':
        """Internal class method that computes the theta integrand term associated with a single Hough function.

        Parameters
        ----------
        action_string : str
            The string that links to a function that retrieves the attribute you want.
        hough_object : HF
            The HF object from which you are retrieving a specific attribute.
        """
        return getattr(cls, action_string).value(hough_object)
