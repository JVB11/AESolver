"""Python module that computes the Hough functions using a Chebyshev collocation method similar to Wang et al. (2016).

Notes
-----
Fit into a python (callable) class by Jordan Van Beeck. Computes the Hough functions in the same way as the original code written by Vincent Prat.

License: GPL-3+
Authors: Vincent Prat <vincent.prat@cea.fr> and Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import logging
import numpy as np
from operator import mul
from functools import partial

# import custom packages
from .hough_instructions import InstructionsEnum  # translation of instructions
from cheby_transform import ChebyTransform  # Chebyshev collocation object

# typing imports
from typing import Any, Callable


# set up logger
logger = logging.getLogger(__name__)


# generic function used to check if value is not None (for properties)
def check_if_none(value: Any, my_func_if_not_none: Callable) -> Any | None:
    """Checks if an input value is not None. If that is the case, call the provided function using the input value.

    Parameters
    ----------
    value : Any
        Value on which a function should be called, if possible. If the value is None, no function is called.
    my_func_if_not_none : Callable
        The function that needs to be called using the input value, if possible.

    Returns
    -------
    Any | None
        Result of the function, when called using the input value, if possible. If the input value is None, this function returns None.
    """
    if value is None:
        return None
    else:
        return my_func_if_not_none(value)


# generic function used to check if 2 values are not None and can be concatenated
def check_concatenate(value1: Any, value2: Any) -> np.ndarray | None:
    """Checks if two input values are not None. If that is the case, they are concatenated into a single array.

    Parameters
    ----------
    value1 : Any
        First value that needs to be concatenated, if possible.
    value2 : Any
        Second value that needs to be concatenated, if possible.

    Returns
    -------
    np.ndarray | None
        The concatenated array for the two input values, or None, if the concatenation cannot be done.
    """
    if value1 is not None and value2 is not None:
        return np.concatenate([value1, value2[::-1]])
    else:
        return None


# class used to compute Hough functions
class HoughFunction:
    """Python class containing methods used to compute the Hough functions based
    on a Chebyshev collocation method.

    Parameters
    ----------
    nu : float | complex
        The spin factor at which we should compute the Hough function.
    l : int
        The degree of the Hough function.
    m : int
        The azimuthal order of the Hough function.
    npts : int, optional
        The total number of points on the Chebyshev collocation grid; by default 200.
    lmbd : None or float or complex, optional
        Optional parameter that specifies the initial guess of the eigenvalue. If None/no parameter passed, this is estimated as -l*(l + 1); by default None.
    use_complex : bool, optional
        If True, allow complex-valued Hough functions (complex valued spin factors). If False, use real-valued Hough functions; by default False.
    definition : str, optional
        Select the definition of the Hough function. Default: 'prat'.
    lmbd_adjust : bool, optional
        Select if you want to adjust the lambda value computed using the collocation scheme; by default True.
    """

    # attribute declaration
    _nu: float
    _l: float
    _m: float
    _parity: int
    _parity_factor: int
    _cheby: ChebyTransform
    _mu: np.ndarray
    _s: np.ndarray
    _lmbd: None | float
    _lmbd_adjust: bool
    _eigenvalue: float | None
    _hr: np.ndarray | None
    _ht: np.ndarray | None
    _hp: np.ndarray | None
    _hrp: np.ndarray | None
    _htp: np.ndarray | None
    _hpp: np.ndarray | None
    _hrpp: np.ndarray | None
    _denom: np.ndarray | None
    _eigvals: np.ndarray | None
    _eigvecs: np.ndarray | None
    _my_instructions: dict

    # initialization method
    def __init__(
        self,
        nu: complex | float,
        l: int,
        m: int,
        npts: int = 200,
        lmbd: float | None = None,
        use_complex: bool = False,
        definition: str = 'prat',
        lmbd_adjust: bool = True,
    ):
        # initialize the spin factor
        self._nu = nu.real  # if use_complex else nu.real
        # initialize the quantum numbers
        self._l, self._m = float(l), float(m)
        # define and store the parity (also called the parity factor)
        self._parity = (l - m) % 2
        self._parity_factor = m % 2
        # initialize the Chebyshev collocation module
        self._cheby = ChebyTransform(npts=npts)
        # make a link with the specific mu and s variables
        self._mu = self._cheby.mu
        self._s = self._cheby.s
        # store initial estimate of the eigenvalue, if provided
        self._lmbd = lmbd if (lmbd is None) or use_complex else lmbd.real
        self._lmbd_adjust = lmbd_adjust
        # initialize parameters that will hold the eigenvalue + Hough functions, as well as their derivatives
        self._eigenvalue = None
        self._hr = None
        self._ht = None
        self._hp = None
        self._hrp = None
        self._htp = None
        self._hpp = None
        self._hrpp = None
        # initialize the denominator attribute used for computations
        self._denom = None
        # initialize the eigenvectors and eigenvalues attributes obtained through collocation (used for computations)
        self._eigvals = None
        self._eigvecs = None
        # select which form of the Hough functions you want to obtain
        self._my_instructions = InstructionsEnum.get_instructions(
            instruction=definition.lower()
        )

    # define some partial function getters
    def _mul_kgrav(self) -> Callable:
        """Get argumentative function for property getters.

        Returns
        -------
        Callable
            Should be run when a getter target is None.
        """
        return partial(mul, (-1.0) ** (abs(self.k_grav)))

    def _mul_kgrav_1(self) -> Callable:
        """Get argumentative function for property getters.

        Returns
        -------
        Callable
            Should be run when a getter target is None.
        """
        return partial(mul, (-1.0) ** (abs(self.k_grav) + 1.0))

    # define the properties - getter methods
    @property
    def l(self) -> int:
        """Return the spherical degree l.

        Returns
        -------
        int
            Return the spherical degree l (some form of that, since we are dealing with combinations of spherical harmonics/Hough functions)
        """
        return int(self._l)

    @property
    def m(self) -> int:
        """Return the azimuthal wavenumber m.

        Returns
        -------
        int
            Return the azimuthal wavenumber m.
        """
        return int(self._m)

    @property
    def mu(self) -> np.ndarray:
        """Return the mu array.

        Returns
        -------
        np.ndarray
            The mu = cos(theta) array.
        """
        return self._mu

    @property
    def thetas(self) -> np.ndarray:
        """Return the theta array.

        Returns
        -------
        np.ndarray
            The theta array.
        """
        return np.arccos(self._mu)

    @property
    def nu(self) -> float:
        """Return the spin parameter.

        Returns
        -------
        float
            The spin parameter nu = 2 * Omega (angular frequency vector of rotation) / omega_corot (pulsation frequency in corotating frame).
        """
        return self._nu

    @property
    def k_grav(self) -> int:
        """Return the mode identification parameter k for gravito-inertial modes (see Lee & Saio, 1997).

        Returns
        -------
        int
            Mode identification parameter k (Lee & Saio, 1997).
        """
        return int(self._l - abs(self._m))

    @property
    def neg_mu_hr(self) -> np.ndarray | None:
        """Return parity-adjusted (based on the radial Hough function) part of the theta-space ranging from mu = -1 to 0.

        Returns
        -------
        np.ndarray | None
            Parity-adjusted (radial Hough function) part of the theta-space ranging from mu = -1 to 0, or None.
        """
        # return the appropriate value
        return check_if_none(
            value=self._hr, my_func_if_not_none=self._mul_kgrav()
        )

    @property
    def neg_mu_ht(self) -> np.ndarray | None:
        """Return parity-adjusted (based on the theta Hough function) part of the theta-space ranging from mu = -1 to 0.

        Returns
        -------
        np.ndarray
            Parity-adjusted (theta Hough function) part of the theta-space ranging from mu = -1 to 0.
        """
        # return the appropriate value
        return check_if_none(
            value=self._ht, my_func_if_not_none=self._mul_kgrav_1()
        )

    @property
    def neg_mu_hp(self) -> np.ndarray | None:
        """Return parity-adjusted (based on the phi Hough function) part of the theta-space ranging from mu = -1 to 0.

        Returns
        -------
        np.ndarray
            Parity-adjusted (phi Hough function) part of the theta-space ranging from mu = -1 to 0.
        """
        # return the appropriate value
        return check_if_none(
            value=self._hp, my_func_if_not_none=self._mul_kgrav()
        )

    @property
    def neg_mu_hrp(self) -> np.ndarray | None:
        """Return parity-adjusted (based on the derivative of the radial Hough function) part of the theta-space ranging from mu = -1 to 0.

        Returns
        -------
        np.ndarray
            Parity-adjusted (derivative of radial Hough function) part of the theta-space ranging from mu = -1 to 0.
        """
        # return the appropriate value
        return check_if_none(
            value=self._hrp, my_func_if_not_none=self._mul_kgrav_1()
        )

    @property
    def neg_mu_hrp_theta(self):
        """Returns parity-adjusted theta derivative of the Radial Hough function in the part of the theta space ranging from mu = -1 to 0.

        Returns
        -------
        np.ndarray
            Parity-adjusted theta derivative of the radial Hough function for mu = -1 to 0.
        """
        # define alternative to None value
        return check_if_none(
            value=self._hrp_theta, my_func_if_not_none=self._mul_kgrav_1()
        )

    @property
    def neg_mu_hrpp(self):
        """Return parity-adjusted (based on the second derivative of the radial Hough function) part of the theta-space ranging from mu = -1 to 0.

        Returns
        -------
        np.ndarray
            Parity-adjusted (second derivative of radial Hough function) part of the theta-space ranging from mu = -1 to 0.
        """
        return check_if_none(
            value=self._hrpp, my_func_if_not_none=self._mul_kgrav_1()
        )

    @property
    def neg_mu_hrpp_theta(self):
        """Returns parity-adjusted second theta derivative of the Radial Hough function in the part of the theta space ranging from mu = -1 to 0.

        Returns
        -------
        np.ndarray
            Parity-adjusted second theta derivative of the radial Hough function for mu = -1 to 0.
        """
        return check_if_none(
            value=self._hrpp_theta, my_func_if_not_none=self._mul_kgrav_1()
        )

    @property
    def neg_mu_htp(self):
        """Return parity-adjusted (based on the derivative of the theta Hough function) part of the theta-space ranging from mu = -1 to 0.

        Returns
        -------
        np.ndarray
            Parity-adjusted (derivative of theta Hough function) part of the theta-space ranging from mu = -1 to 0.
        """
        return check_if_none(
            value=self._htp, my_func_if_not_none=self._mul_kgrav()
        )

    @property
    def neg_mu_htp_theta(self):
        """Returns parity-adjusted theta derivative of the theta Hough function in the part of the theta space ranging from mu = -1 to 0.

        Returns
        -------
        np.ndarray
            Parity-adjusted theta derivative of the theta Hough function for mu = -1 to 0.
        """
        return check_if_none(
            value=self._htp_theta, my_func_if_not_none=self._mul_kgrav()
        )

    @property
    def neg_mu_hpp(self):
        """Return parity-adjusted (based on the derivative of the phi Hough function) part of the theta-space ranging from mu = -1 to 0.

        Returns
        -------
        np.ndarray
            Parity-adjusted (derivative of phi Hough function) part of the theta-space ranging from mu = -1 to 0.
        """
        return check_if_none(
            value=self._hpp, my_func_if_not_none=self._mul_kgrav_1()
        )

    @property
    def neg_mu_hpp_theta(self):
        """Returns parity-adjusted theta derivative of the phi Hough function in the part of the theta space ranging from mu = -1 to 0.

        Returns
        -------
        np.ndarray
            Parity-adjusted theta derivative of the phi Hough function for mu = -1 to 0.
        """
        return check_if_none(
            value=self._hpp_theta, my_func_if_not_none=self._mul_kgrav_1()
        )

    @property
    def neg_mu(self):
        """Return part of the theta-space ranging from mu = -1 to 0.

        Returns
        -------
        np.ndarray
            Part of the theta-space ranging from mu = -1 to 0.
        """
        return (-1.0) * self._mu

    @property
    def neg_thetas(self):
        """Return part of the theta array ranging from mu = -1 to 0.

        Returns
        -------
        np.ndarray
            Part of the theta-space ranging from mu = -1 to 0.
        """
        return np.arccos(self.neg_mu)

    @property
    def mu_range(self):
        """Return entire theta space in terms of mu = cos(theta).

        Returns
        -------
        np.ndarray
            Entire theta space in terms of mu.
        """
        return np.concatenate([self.neg_mu, self.mu[::-1]])

    @property
    def thetas_range(self):
        """Return entire theta space.

        Returns
        -------
        np.ndarray
            Entire theta space.
        """
        return np.arccos(self.mu_range)

    @property
    def hr_range(self):
        """Return parity-adjusted (based on the radial Hough function) theta space in terms of mu = cos(theta).

        Returns
        -------
        np.ndarray
            Parity-adjusted (radial Hough function) theta space in terms of mu = cos(theta).
        """
        return check_concatenate(value1=self.neg_mu_hr, value2=self._hr)

    @property
    def hr_range_m(self):
        """Return parity-adjusted (based on the radial Hough function) theta space in terms of mu = cos(theta) times m.

        Returns
        -------
        np.ndarray
            Parity-adjusted (radial Hough function) theta space in terms of mu = cos(theta) times m.
        """
        return check_if_none(
            value=self.hr_range, my_func_if_not_none=partial(mul, -1.0 * self.m)
        )
        # (NOTE the negative sign for the m-term, because of different definitions of 'm' in Lee(2012) and Prat et al. (2019))

    @property
    def ht_range(self):
        """Return parity-adjusted (based on the theta Hough function) theta space in terms of mu = cos(theta).

        Returns
        -------
        np.ndarray
            Parity-adjusted (theta Hough function) theta space in terms of mu = cos(theta).
        """
        return check_concatenate(value1=self.neg_mu_ht, value2=self._ht)

    @property
    def hp_range(self):
        """Return parity-adjusted (based on the phi Hough function) theta space in terms of mu = cos(theta).

        Returns
        -------
        np.ndarray
            Parity-adjusted (phi Hough function) theta space in terms of mu = cos(theta).
        """
        return check_concatenate(value1=self.neg_mu_hp, value2=self._hp)

    @property
    def hrp_range(self):
        """Return parity-adjusted (based on the derivative of the radial Hough function) theta space in terms of mu = cos(theta).

        Returns
        -------
        np.ndarray
            Parity-adjusted (derivative of radial Hough function) theta space in terms of mu = cos(theta).
        """
        return check_concatenate(value1=self.neg_mu_hrp, value2=self._hrp)

    @property
    def hrp_range_theta(self):
        """Return parity-adjusted theta derivative of the radial Hough function, at the locations of the supplied mu = cos(theta) array.

        Returns
        -------
        np.ndarray
            Parity-adjusted theta derivative of the Radial Hough function.
        """
        return check_concatenate(
            value1=self.neg_mu_hrp_theta, value2=self._hrp_theta
        )

    @property
    def hrpp_range(self):
        """Return parity-adjusted (based on the second derivative of the radial Hough function) theta space in terms of mu = cos(theta).

        Returns
        -------
        np.ndarray
            Parity-adjusted (second derivative of radial Hough function) theta space in terms of mu = cos(theta).
        """
        return check_concatenate(value1=self.neg_mu_hrpp, value2=self._hrpp)

    @property
    def hrpp_range_theta(self):
        """Return parity-adjusted second theta derivative of the radial Hough function, at the locations of the supplied mu = cos(theta) array.

        Returns
        -------
        np.ndarray
            Parity-adjusted second theta derivative of the Radial Hough function.
        """
        return check_concatenate(
            value1=self.neg_mu_hrpp_theta, value2=self._hrpp_theta
        )

    @property
    def htp_range(self):
        """Return parity-adjusted (based on the derivative of the theta Hough function) theta space in terms of mu = cos(theta).

        Returns
        -------
        np.ndarray
            Parity-adjusted (derivative of theta Hough function) theta space in terms of mu = cos(theta).
        """
        return check_concatenate(value1=self.neg_mu_htp, value2=self._htp)

    @property
    def htp_range_theta(self):
        """Return parity-adjusted theta derivative of the theta Hough function, at the locations of the supplied mu = cos(theta) array.

        Returns
        -------
        np.ndarray
            Parity-adjusted theta derivative of the theta Hough function.
        """
        return check_concatenate(
            value1=self.neg_mu_htp_theta, value2=self._htp_theta
        )

    @property
    def hpp_range(self):
        """Return parity-adjusted (based on the derivative of the phi Hough function) theta space in terms of mu = cos(theta).

        Returns
        -------
        np.ndarray
            Parity-adjusted (derivative of phi Hough function) theta space in terms of mu = cos(theta).
        """
        return check_concatenate(value1=self.neg_mu_hpp, value2=self._hpp)

    @property
    def hpp_range_theta(self):
        """Return parity-adjusted theta derivative of the phi Hough function, at the locations of the supplied mu = cos(theta) array.

        Returns
        -------
        np.ndarray
            Parity-adjusted theta derivative of the phi Hough function.
        """
        return check_concatenate(
            value1=self.neg_mu_hpp_theta, value2=self._hpp_theta
        )

    @property
    def theta_phi_hough_tilde(self):
        """Return the values of the theta_phi term defined in Eqn. B16 of Lee(2012).

        Returns
        -------
        np.ndarray
            Theta_phi term defined in Eqn. B16 of Lee(2012).
        """
        if (
            self._ht is not None
            and self._hp is not None
            and self.neg_mu_ht is not None
            and self.neg_mu_hp is not None
        ):
            # get positive mu part (NOTE the negative sign for the m-term, because of different definitions of 'm' in Lee(2012) and Prat et al. (2019))
            pos_mu_part = ((self._m * self._ht) - (self._mu * self._hp))[::-1]
            # get negative mu part (NOTE the negative sign for the m-term, because of different definitions of 'm' in Lee(2012) and Prat et al. (2019))
            neg_mu_part = (self._m * self.neg_mu_ht) - (
                self.neg_mu * self.neg_mu_hp
            )
            return np.concatenate([neg_mu_part, pos_mu_part])
        else:
            return None

    @property
    def phi_phi_hough_tilde(self):
        """Return the values of the phi_phi term defined in Eqn. B16 of Lee(2012).

        Returns
        -------
        np.ndarray
            Phi_phi term defined in Eqn. B16 of Lee(2012).
        """
        if (
            self._ht is not None
            and self._hp is not None
            and self.neg_mu_ht is not None
            and self.neg_mu_hp is not None
        ):
            # get positive mu part (NOTE the positive sign for the m-term, because of different definitions of 'm' in Lee(2012) and Prat et al. (2019))
            pos_mu_part = (
                ((self._mu * self._ht) - (self._m * self._hp)) / self._s
            )[::-1]
            # get negative mu part (NOTE the positive sign for the m-term, because of different definitions of 'm' in Lee(2012) and Prat et al. (2019))
            neg_mu_part = (
                (self.neg_mu * self.neg_mu_ht) - (self._m * self.neg_mu_hp)
            ) / self._s
            # return complete part
            return np.concatenate([neg_mu_part, pos_mu_part])
        else:
            return None

    # make the class a callable
    def __call__(self):
        """Callable used to retrieve and store the specific eigenvalue and Hough functions + derivatives.

        Parameters
        ----------
        get_lee_saio_def : bool, optional
            If True, obtain the Lee & Saio (1997) azimuthal Hough function defined in their equation (29). If False, obtain the Prat et al. (2019) azimuthal Hough function defined in their equation (A.3). By default False.
        """
        # perform collocation
        self._colloc()
        # retrieve radial Hough function and eigenvalue
        self._get_eigenvalue_and_radial_hough()
        # retrieve latitudinal Hough function
        self._get_latitudinal_hough()
        # retrieve azimuthal Hough function
        self._get_azimuthal_hough()
        # get the derivatives
        self._get_derivatives()
        # get the theta derivatives
        self._get_theta_derivatives()

    # method used to solve Laplace tidal equations for the radial Hough function
    # (see appendix A, Prat et al. (2019))
    def _colloc(self):
        """Method that performs the collocation in order to solve the Laplace tidal equations for the radial Hough function (see Prat et al. (2019))."""
        # define the coefficients of the Laplace tidal equations
        self._denom = 1.0 - (self._nu**2.0 * self._mu**2.0)  # denominator
        _coeffs_2 = self._s**2.0 / self._denom  # 2nd order
        _coeffs_1 = -2.0 * self._mu * (1.0 - self._nu**2.0) / self._denom**2.0
        _coeffs_0 = self._nu * self._m * (
            1.0 + self._nu**2.0 * self._mu**2.0
        ) / (self._denom**2.0) - self._m**2.0 / (self._s**2 * self._denom)
        # solve the Laplace tidal equations
        self._eigvals, self._eigvecs = self._cheby.solve_eig(
            _coeffs_0, _coeffs_1, _coeffs_2, self._parity, self._parity_factor
        )

    # method used to remove complex and positive eigenvalues + locate the
    # fundamental solution
    def _get_eigenvalue_and_radial_hough(self):
        """Method that locates the fundamental solution to the Laplace tidal equations."""
        if self._eigvals is not None and self._eigvecs is not None:
            # remove complex eigenvalues
            _mask = self._eigvals.imag == 0
            self._eigvals = self._eigvals[_mask].real
            self._eigvecs = self._eigvecs[:, _mask].real
            # remove positive eigenvalues
            _mask = self._eigvals < 0
            self._eigvals = self._eigvals[_mask]
            self._eigvecs = self._eigvecs[:, _mask]
            # sort eigenvalues and vectors
            _mask = np.argsort(self._eigvals)
            self._eigvals = self._eigvals[_mask]
            self._eigvecs = self._eigvecs[:, _mask]
            # if estimate for eigenvalue was not provided, generate automatic estimate
            if self._lmbd is None:
                self._lmbd = -self._l * (self._l + 1.0)
            # if list estimate is provided (list containing copies of same value(!)), use the first value (NOTE the minus sign due to conflicting definitions of the  Laplace tidal equations in Prat et al. (2019) and Lee & Saio (1997) / GYRE!)
            elif isinstance(self._lmbd, list):
                # if input value = from GYRE, self._lmbd_adjust should be TRUE
                # if input value is inverted in sign self._lmbd_adjust should be FALSE
                self._lmbd = (
                    (-1.0) * self._lmbd[0]
                    if self._lmbd_adjust
                    else self._lmbd[0]
                )
            # other case: float value (NOTE the minus sign due to conflicting definitions of the Laplace tidal equations in Prat et al. (2019) and Lee & Saio (1997) / GYRE!)
            else:
                # adjust input value when necessary (see comments above)
                if self._lmbd_adjust:
                    self._lmbd *= -1.0
            # get first occurrence of smallest distance to calculated eigenvalue
            _mask = np.argmin(np.abs(self._lmbd - self._eigvals))
            # obtain the eigenvalue and radial Hough function
            self._eigenvalue = self._eigvals[_mask]
            self._hr = self._eigvecs[:, _mask]
            # if last point is negative, switch sign
            if self._hr[-1] < 0.0:
                self._hr *= -1.0
            # NORMALIZE the radial Hough function (according to Prat et al. 2019)
            self._hr /= np.abs(self._hr).max()
        else:
            self._hr = None
            self._lmbd = None
            self._eigenvalue = None

    # methods used to compute the latitudinal and azimuthal Hough functions
    def _get_latitudinal_hough(self):
        """Method that computes the latitudinal Hough function."""
        if self._hr is not None and self._denom is not None:
            # compute the coefficients. (see Appendix A (Prat et al. 2019) where Hr' denotes a derivative with respect to theta, which has to be converted to a derivative with respect to mu (=cos(theta)) in order to obtain the expressions for the coefficients below)
            _coeffs_1_ht = -(self._s**2.0) / self._denom
            _coeffs_0_ht = -self._m * self._nu * self._mu / self._denom
            # compute ht (latitudinal Hough function) from hr
            self._ht = (
                self._cheby.apply(
                    self._hr,
                    _coeffs_0_ht,
                    _coeffs_1_ht,
                    None,
                    self._parity,
                    self._parity_factor,
                )
                / self._s
            )
            # adjust form if necessary (GYRE definition)
            if self._my_instructions['sin_factor']:
                self._ht *= self._s
        else:
            self._ht = None

    def _get_azimuthal_hough(self):
        """Method that computes the azimuthal Hough function.

        Parameters
        ----------
        get_lee_saio_def : bool, optional
            If True, obtain the Lee & Saio (1997) azimuthal Hough function defined in their equation (29). If False, obtain the Prat et al. (2019) azimuthal Hough function defined in their equation (A.3). By default False.
        """
        if self._hr is not None and self._denom is not None:
            # compute the coefficients. (see Appendix A (Prat et al. 2019) where Hr' denotes a derivative with respect to theta, which has to be converted to a derivative with respect to mu (=cos(theta)) in order to obtain the expressions for the coefficients below)
            _coeffs_1_hp = self._nu * self._mu * self._s**2.0 / self._denom
            _coeffs_0_hp = self._m / self._denom
            # compute hp (azimuthal Hough function) from hr
            self._hp = (
                self._cheby.apply(
                    self._hr,
                    _coeffs_0_hp,
                    _coeffs_1_hp,
                    None,
                    self._parity,
                    self._parity_factor,
                )
                / self._s
            )
            # multiplicative factor if Lee & Saio def. is to be followed.
            if self._my_instructions['adjust_phi_sign']:
                self._hp *= -1.0
            # adjust form if necessary (GYRE definition)
            if self._my_instructions['sin_factor']:
                self._hp *= self._s
        else:
            self._hp = None

    # method used to compute the derivatives of the Hough functions
    def _get_derivatives(self):
        """Method that computes the (mu) derivatives of the Hough functions."""
        if self._hr is not None and self._denom is not None:
            # compute the derivative of Hr
            self._hrp = self._cheby.diff1(
                self._hr, self._parity, self._parity_factor
            )
            # compute the second derivative of Hr
            self._hrpp = self._cheby.diff2(
                self._hr, self._parity, self._parity_factor
            )
            # compute the derivative of Ht
            _coeffs_2_htp = -(self._s**4.0) / self._denom
            _coeffs_1_htp = (
                self._s**2.0
                * (
                    -self._m * self._nu * self._mu * self._denom
                    + self._mu
                    + self._mu**3.0 * self._nu**2.0
                    - 2.0 * self._mu * self._nu**2.0
                )
                / self._denom**2.0
            )
            _coeffs_0_htp = (
                self._m
                * self._nu
                * (
                    2.0 * self._mu**4.0 * self._nu**2.0
                    - self._mu**2.0 * self._nu**2.0
                    - 1.0
                )
                / self._denom**2.0
            )
            self._htp = (
                self._cheby.apply(
                    self._hr,
                    _coeffs_0_htp,
                    _coeffs_1_htp,
                    _coeffs_2_htp,
                    self._parity,
                    self._parity_factor,
                )
                / self._s**3.0
            )
            # adjust form if necessary (GYRE definition)
            if self._my_instructions['sin_factor']:
                self._htp = (self._s * self._htp) - (
                    (self._mu * self._ht) / self._s ** (2.0)
                )
            # compute the derivative of Hp
            _coeffs_2_hpp = self._nu * self._mu * self._s**4.0 / self._denom
            _coeffs_1_hpp = (
                self._s**2.0
                * (
                    self._m * self._denom
                    + self._nu
                    + self._nu**3.0 * self._mu**2.0
                    - 2.0 * self._mu**2.0 * self._nu
                )
                / self._denom**2.0
            )
            _coeffs_0_hpp = (
                self._m
                * self._mu
                * (
                    1.0
                    - self._nu**2.0 * self._mu**2.0
                    + 2.0 * self._nu**2.0
                    - 2.0 * self._nu**2.0 * self._mu**2.0
                )
                / self._denom**2.0
            )
            self._hpp = (
                self._cheby.apply(
                    self._hr,
                    _coeffs_0_hpp,
                    _coeffs_1_hpp,
                    _coeffs_2_hpp,
                    self._parity,
                    self._parity_factor,
                )
                / self._s**3.0
            )
            # adjust form if necessary (GYRE definition)
            if self._my_instructions['sin_factor']:
                self._hpp = (self._s * self._hpp) - (
                    (self._mu * self._hp) / self._s ** (2.0)
                )
        else:
            self._hpp = None
            self._htp = None
            self._hrp = None
            self._hrpp = None

    # method used to compute the theta derivatives
    def _get_theta_derivatives(self):
        """Method that computes the theta derivatives of the Hough functions, in terms of the mu-derivative equivalents."""
        # compute the first theta derivatives
        self._hrp_theta = (
            -self._s * self._hrp if self._hrp is not None else None
        )
        self._htp_theta = (
            -self._s * self._htp if self._htp is not None else None
        )
        self._hpp_theta = (
            -self._s * self._hpp if self._hpp is not None else None
        )
        # compute the second theta derivatives
        if self._hrpp is None:
            self._hrpp_theta = None
        else:
            self._hrpp_theta = self._s**2.0 * self._hrpp - self._mu * self._hrp
