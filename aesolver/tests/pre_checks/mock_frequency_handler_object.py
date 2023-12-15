'''Python module defining a mock frequency handler object to be used during testing of the aesolver.pre_checks module.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
'''
# import statements
import pytest
import numpy as np


# define the mock frequency handler class
class MockFreqHandler:
    """Mock frequency handler object to be used during testing of the aesolver.pre_checks module."""
    
    # attribute type declarations
    driving_rates: np.ndarray
    q_factor: np.ndarray
    fail: bool
    reason: str | None
    parametric_check: str | None
    direct_check: str | None
    driven_check: str | None
    two_mode_harmonic: bool
    
    def __init__(self, driving_rates: np.ndarray,
                 q_factor: np.ndarray, fail: bool,
                 reason: str | None,
                 parametric_check: str | None,
                 direct_check: str | None,
                 driven_check: str | None,
                 two_mode_harmonic: bool) -> None:
        # store the mock data
        self.driving_rates = driving_rates
        self.q_factor = q_factor
        self.fail = fail
        self.reason = reason
        self.parametric_check = parametric_check
        self.direct_check = direct_check
        self.driven_check = driven_check
        self.two_mode_harmonic = two_mode_harmonic

    def return_parametric_check_info(self) -> pytest.param:
        """Returns useful information for the parametric check method.
        
        Returns
        -------
        pytest.param
            The pytest parameter for the parametric check method.
        """
        if self.parametric_check is None:
            return pytest.param('PC')
        else:
            return pytest.param(
                'PC', marks=pytest.mark.xfail(
                    reason=self.parametric_check
                    )
                )

    def return_direct_check_info(self) -> pytest.param:
        """Returns useful information for the direct check method.
        
        Returns
        -------
        pytest.param
            The pytest parameter for the direct check method.
        """
        if self.direct_check is None:
            return pytest.param('DC')
        else:
            return pytest.param(
                'DC', marks=pytest.mark.xfail(
                    reason=self.direct_check
                    )
                )
            
    def return_driven_check_info(self) -> pytest.param:
        """Returns useful information for the parametric check method.
        
        Returns
        -------
        pytest.param
            The pytest parameter for the parametric check method.
        """
        if self.driven_check is None:
            return pytest.param('DRV')
        else:
            return pytest.param(
                'DRV', marks=pytest.mark.xfail(
                    reason=self.driven_check
                    )
                )

    def return_param(self) -> pytest.param:
        """Returns a pytest parameter containing an xfail, if necessary.
        
        Returns
        -------
        pytest.param
            The pytest parameter representing this class. Has an xfail mark if necessary.
        """
        if self.fail:
            return pytest.param(
                'FH',
                marks=pytest.mark.xfail(reason=self.reason)
                )
        else:
            return pytest.param('FH')
