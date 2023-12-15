'''Python module used to create a mock object that mimics the quadratic coupling coefficient solver object.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
'''
# import statements
import typing
import numpy as np
# import support module used to subclass
from mock_solver_object_generic import \
    MockSolverGeneric as MSG
# import pytest util function
from pytest_util_functions import check_assertions


# define the mock coupler object
class MockCoupler:
    """Defines the mock coupler object.
    
    Parameters
    ----------
    azimuthal_wavenumbers: list[int] | None, optional
        Stores the azimuthal wavenumbers. If None, default to [2, 2, 2]; by default None.
    nr_vals: int, optional
        The number of values in the data arrays; by default 5.
    cc_val: float, optional
        The coupling coefficient value; by default 2.0.
    hr_values: list[np.ndarray] | None, optional
        The list of radial Hough function values or None. If None, the radial Hough function values are all set equal to 1; by default None.
    """
    
    # attribute type declarations
    eta_lee_coupling_coefficient: float
    _mu_values: np.ndarray
    _hr_mode_1: np.ndarray
    _hr_mode_2: np.ndarray
    _hr_mode_3: np.ndarray
    _m_mode_1: int
    _m_mode_2: int
    _m_mode_3: int
    
    def __init__(self,
        azimuthal_wavenumbers: list[int]| None=None,
        nr_vals: int=5, cc_val: float=2.0,
        hr_values: list[np.ndarray] | None=None
        ) -> None:
        # initialize the coupling coefficient
        self.eta_lee_coupling_coefficient = cc_val
        # assert that the number of values in
        # the arrays is at least 2
        assert nr_vals >= 2
        # initialize the mu values array
        self._mu_values = np.linspace(
            -1.0, 1.0, num=nr_vals,
            dtype=np.float64
            )
        # assert the list of hr values and its elements
        # are of the expected size and type
        if hr_values:
            check_assertions(
                my_val=hr_values, top_length=3,
                element_type=np.ndarray,
                element_length=nr_vals
                )
        else:
            hr_values = [np.ones_like(self._mu_values)] * 3
        # assert that the azimuthal wavenumber list input
        # is of the correct length and its elements are of
        # the correct type
        if azimuthal_wavenumbers:
            check_assertions(
                my_val=azimuthal_wavenumbers,
                top_length=3, element_type=int
                )
        else:
            azimuthal_wavenumbers = [2, 2, 2]
        # initialize the radial Hough mode arrays
        # and the azimuthal wave numbers
        for i in range(1, 4):
            setattr(self, f"_hr_mode_{i}", 
                    hr_values[i - 1])             
            setattr(self, f"_m_mode_{i}",
                    azimuthal_wavenumbers[i - 1])


# define the extended mock solver object
class MockSolverThreeModeDefault(MSG):
    """Subclass of the generic mock solver object used to test the three mode stationary solving module of aesolver."""
    
    # attribute type declarations
    _coupler: MockCoupler  # redefined from the superclass
    
    def __init__(self, nr_modes: int = 3,
                 azimuthal_wavenumbers: list[int] | None=None,
                 nr_vals: int=5, cc_val: float=2.0,
                 hr_values: list[np.ndarray] | None=None) -> None:
        # initialize the superclass
        super().__init__(nr_modes=nr_modes)
        # overwrite the coupler object
        self._coupler = MockCoupler(
            azimuthal_wavenumbers=azimuthal_wavenumbers,
            nr_vals=nr_vals, cc_val=cc_val,
            hr_values=hr_values
            )
