'''Pytest module used to test the hyperbolicity-checking module.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
'''
# import statements
import pytest
import sys
import numpy as np
# import pytest util class
from pytest_util_classes import EVs
# import functionalities to be tested
from aesolver.stability_checker import \
    hyper_jac, check_hyper, hyper_jac_three


# INPUT VALUES


# store the parametric input values for the test
class InValsParametric(EVs):
    """Parametric Input values for the hyperbolicity checker tests."""
    
    GAMMAS = (1e-9 * np.array([1.0, -2.5, -3.0])).tobytes()
    GAMMAS_THREE = (1e-9 * np.array([2.0, -1.75, -1.75])).tobytes()
    OMEGAS = (1e-7 * np.array([1.0, 2.0, 3.0])).tobytes()
    OMEGAS_THREE = (1e-7 * np.array([2.0, 1.75, 1.75])).tobytes()
    Q_VAL = 10**2.0
    ETA_VAL = 10**7.0
    
    
# store the driven input values for the test
class InValsDriven(EVs):
    """Driven Failing Input values for the hyperbolicity checker tests."""
    
    GAMMAS = (1e-9 * np.array([1.0, 2.5, 3.0])).tobytes()
    OMEGAS = (1e-7 * np.array([1.0, 2.0, 3.0])).tobytes()
    GAMMAS_THREE = (1e-9 * np.array([2.0, 1.75, 1.75])).tobytes()
    OMEGAS_THREE = (1e-7 * np.array([2.0, 1.75, 1.75])).tobytes()
    Q_VAL = 10**2.0
    ETA_VAL = 10**7.0


# store the direct input values for the test
class InValsDirect(EVs):
    """Direct Failing Input values for the hyperbolicity checker tests."""
    
    GAMMAS = (1e-9 * np.array([-1.0, 2.5, 3.0])).tobytes()
    OMEGAS = (1e-7 * np.array([1.0, 2.0, 3.0])).tobytes()
    GAMMAS_THREE = (1e-9 * np.array([-2.0, 1.75, 1.75])).tobytes()
    OMEGAS_THREE = (1e-7 * np.array([2.0, 1.75, 1.75])).tobytes()
    Q_VAL = 10**2.0
    ETA_VAL = 10**7.0
    
    
# store the damped input values for the test
class InValsDamped(EVs):
    """Damped Failing Input values for the hyperbolicity checker tests."""
    
    GAMMAS = (1e-9 * np.array([-1.0, -2.5, -3.0])).tobytes()
    OMEGAS = (1e-7 * np.array([1.0, 2.0, 3.0])).tobytes()
    GAMMAS_THREE = (1e-9 * np.array([-2.0, -1.75, -1.75])).tobytes()
    OMEGAS_THREE = (1e-7 * np.array([2.0, 1.75, 1.75])).tobytes()
    Q_VAL = 10**2.0
    ETA_VAL = 10**7.0


# store the output values for the test
class OutValsParametric(EVs):
    """Output values for the hyperbolicity checker tests."""
    
    MY_JAC = np.array(
        [[1.0e-09, 1.11803399e-09, 1.0e-09, 5.59044945e-15],
         [-2.23606798e-09, -2.5e-09, -2.23606798e-09, -1.25006250e-14],
         [-3.0e-09, -3.35410197e-09, -3.0e-09, -1.67713483e-14],
         [-1.16269721e+01, 9.99950004e-01, 2.68314742e+00, -4.5e-09]]
        ).tobytes()
    MY_HYP = True
    MY_JAC_THREE = np.array(
        [[2.0e-09, 2.0e-09, -1.00005000e-14],
         [-2.47487373e-09, -1.75e-09, -1.23749874e-14],
         [1.09994500e+01, 5.65657143e+00, -1.5e-09]]
        ).tobytes()
    MY_HYP_THREE = True


# store the output values for the test
class OutValsDriven(EVs):
    """Output values for the hyperbolicity checker tests."""
    
    MY_JAC = np.array(
        [[1.0e-09, np.NaN, np.NaN, 5.59044945e-15],
         [np.NaN, 2.5e-09, -2.23606798e-09, np.NaN],
         [np.NaN, -3.35410197e-09, 3.0e-09, np.NaN],
         [8.04944226e+00, np.NaN,
          np.NaN, 6.0e-09]]
        ).tobytes()
    MY_HYP = False
    MY_JAC_THREE = np.array(
        [[2.0e-09, np.NaN, 1.00005000e-14],
         [np.NaN, 1.75e-09, np.NaN],
         [2.99985001e+00, np.NaN, 5.5e-09]]
        ).tobytes()
    MY_HYP_THREE = False
    

# store the output values for the test
class OutValsDirect(EVs):
    """Output values for the hyperbolicity checker tests."""
    MY_JAC = np.array(
        [[-1.0e-09, 1.11803399e-09, 1.0e-09, -5.59044945e-15],
         [-2.23606798e-09, 2.5e-09, -2.23606798e-09, 1.25006250e-14],
         [-3.0e-09, -3.35410197e-09, 3.0e-09, 1.67713483e-14],
         [1.16269721e+01, -9.99950004e-01,
          -2.68314742e+00, 4.0e-09]]
        ).tobytes()
    MY_HYP = True
    MY_JAC_THREE = np.array(
        [[-2.0e-09, 2.0e-09, -1.00005000e-14],
         [-2.47487373e-09, 1.75e-09, 1.23749874e-14],
         [1.09994500e+01, 5.65657143e+00, 1.5e-09]]
        ).tobytes()
    MY_HYP_THREE = True


# store the output values for the test
class OutValsDamped(EVs):
    """Output values for the hyperbolicity checker tests."""
    MY_JAC = np.array(
        [[-1.0e-09, np.NaN, np.NaN, -5.59044945e-15],
         [np.NaN, -2.5e-09, -2.23606798e-09, np.NaN],
         [np.NaN, -3.35410197e-09, -3.0e-09, np.NaN],
         [-8.04944226e+00, np.NaN, np.NaN, -6.5e-09]]
        ).tobytes()
    MY_HYP = False
    MY_JAC_THREE = np.array(
        [[-2.0e-09, np.NaN, 1.00005000e-14],
         [np.NaN, -1.75e-09, np.NaN],
         [2.99985001e+00, np.NaN, -5.5e-09]]
        ).tobytes()
    MY_HYP_THREE = False
        
        
# COMPUTE TESTS    


# set up a fixture for the test class
@pytest.fixture(
    scope="class",
    params=[
        ("parametric",), ("driven",),
        ("direct",), ("damped",)
        ]
    )
def setup_test_class(request) -> None:
    """Parametrized setup function for the test class that handles cases where coupling coefficients were computed. """
    # get the in and output objects
    _my_in_obj = getattr(
        sys.modules[__name__],
        f"InVals{request.param[0].capitalize()}"
        )
    _my_out_obj = getattr(
        sys.modules[__name__],
        f"OutVals{request.param[0].capitalize()}"
        )
    # store the input values
    request.cls.gammas: np.ndarray = \
        _my_in_obj.get_value("gammas")
    request.cls.omegas: np.ndarray = \
        _my_in_obj.get_value("omegas")
    request.cls.gammas_three: np.ndarray = \
        _my_in_obj.get_value("gammas_three")
    request.cls.omegas_three: np.ndarray = \
        _my_in_obj.get_value("omegas_three")
    request.cls.q_val: np.float64 = \
        _my_in_obj.get_value("q_val")
    request.cls.eta_val: np.float64 = \
        _my_in_obj.get_value("eta_val")
    # store the expected output values
    request.cls.expect_jac: np.ndarray = \
        _my_out_obj.get_value(
            "my_jac"
            ).reshape((4, 4))
    request.cls.expect_hyp: bool = \
        _my_out_obj.get_value("my_hyp")
    request.cls.expect_jac_three: np.ndarray = \
        _my_out_obj.get_value(
            "my_jac_three"
            ).reshape((3,3))
    request.cls.expect_hyp_three: bool = \
        _my_out_obj.get_value("my_hyp_three")
        
        
# define the test class
@pytest.mark.usefixtures('setup_test_class')
class TestHyperBolicityCheck:
    """Tests the hyperbolicity-checking functionality."""
    
    # attribute type declarations
    gammas: np.ndarray
    gammas_three: np.ndarray
    omegas: np.ndarray
    omegas_three: np.ndarray
    q_val: np.float64
    eta_val: np.float64
    expect_jac: np.ndarray
    expect_hyp: bool
    expect_jac_three: np.ndarray
    expect_hyp_three: bool
    
    # test jacobian-computing function
    def test_jac_computer(self):
        """Tests the Jacobian matrix-computing
        functionality.
        """
        # compute the Jacobian matrix
        _my_jac = hyper_jac(
            gammas=self.gammas,
            omegas=self.omegas,
            q=self.q_val,
            eta=self.eta_val
            )
        # get mask for NaN values before matching
        _mm = ~(np.isnan(_my_jac)
                | np.isnan(self.expect_jac))
        # assert the non-NaN Jacobian values are OK
        assert np.allclose(
            _my_jac[_mm], self.expect_jac[_mm]
            )

    # test jacobian-computing function for three modes
    def test_jac_computer_three(self):
        """Tests the Jacobian matrix-computing functionality for a harmonic AE system."""
        # compute the Jacobian matrix
        _my_jac = hyper_jac_three(
            gammas=self.gammas_three,
            omegas=self.omegas_three,
            q=self.q_val,
            eta=self.eta_val
            )
        # get mask for NaN values before matching
        _mm = ~(np.isnan(_my_jac)
                | np.isnan(self.expect_jac_three))
        # assert the non-NaN Jacobian values are OK
        assert np.allclose(
            _my_jac[_mm], self.expect_jac_three[_mm]
            )

    # test whether the correct hyperbolicity
    # check result is obtained
    def test_check_result(self):
        """Tests the hyperbolicity-checking functionality."""
        # compute the Jacobian matrix
        _my_jac = hyper_jac(
            gammas=self.gammas,
            omegas=self.omegas,
            q=self.q_val,
            eta=self.eta_val
            )
        # obtain whether it is hyperbolic
        _my_hyp = check_hyper(
            my_jac=_my_jac
            )
        # assert the hyperbolicity value is OK
        assert _my_hyp == self.expect_hyp        

    # test whether the correct hyperbolicity
    # check result is obtained for three modes
    def test_check_result_three(self):
        """Tests the hyperbolicity-checking functionality for a harmonic AE system."""
        # compute the Jacobian matrix
        _my_jac = hyper_jac_three(
            gammas=self.gammas_three,
            omegas=self.omegas_three,
            q=self.q_val,
            eta=self.eta_val
            )
        # obtain whether it is hyperbolic
        _my_hyp = check_hyper(
            my_jac=_my_jac
            )
        # assert the hyperbolicity value is OK
        assert _my_hyp == self.expect_hyp_three


# NO COMPUTE TESTS


# store input values for a no-compute scenario
# (i.e. precheck not satisfied)
class InValsNoCompute(EVs):
    """No-compute Input values for the hyperbolicity checker tests."""
    
    GAMMAS = (1e-9 * np.array([1.0, -2.5, -3.0])).tobytes()
    GAMMAS_THREE = (1e-9 * np.array([2.0, -1.75, -1.75])).tobytes()
    OMEGAS = (1e-7 * np.array([1.0, 2.0, 3.0])).tobytes()
    OMEGAS_THREE = (1e-7 * np.array([2.0, 1.75, 1.75])).tobytes()
    Q_VAL = 0.0
    ETA_VAL = 0.0
    
    
# store output values for a no-compute scenario
# (i.e. precheck not satisfied)
class OutValsNoCompute(EVs):
    """No-compute output values for the hyperbolicity checker tests."""
    
    MY_JAC = np.array(
        [[1.0e-09, 1.11803399e-09, 1.0e-09, 5.59044945e-15],
         [-2.23606798e-09, -2.5e-09, -2.23606798e-09, -1.25006250e-14],
         [-3.0e-09, -3.35410197e-09, -3.0e-09, -1.67713483e-14],
         [-1.16269721e+01, 9.99950004e-01, 2.68314742e+00, -4.5e-09]]
        ).tobytes()
    MY_HYP = False
    MY_JAC_THREE = np.array(
        [[2.0e-09, 2.0e-09, -1.00005000e-14],
         [-2.47487373e-09, -1.75e-09, -1.23749874e-14],
         [1.09994500e+01, 5.65657143e+00, -1.5e-09]]
        ).tobytes()
    MY_HYP_THREE = False

 
# set up a fixture for the no compute test class
@pytest.fixture(scope="class")
def setup_test_class_no_compute(request) -> None:
    """Setup function for the test class that handles the no-compute case."""
    # get the input object
    _my_in_obj = getattr(
        sys.modules[__name__],
        "InValsNoCompute"
        )
    _my_out_obj = getattr(
        sys.modules[__name__],
        "OutValsNoCompute"
        )
    # store the input values
    request.cls.gammas: np.ndarray = \
        _my_in_obj.get_value("gammas")
    request.cls.omegas: np.ndarray = \
        _my_in_obj.get_value("omegas")
    request.cls.gammas_three: np.ndarray = \
        _my_in_obj.get_value("gammas_three")
    request.cls.omegas_three: np.ndarray = \
        _my_in_obj.get_value("omegas_three")
    request.cls.q_val: np.float64 = \
        _my_in_obj.get_value("q_val")
    request.cls.eta_val: np.float64 = \
        _my_in_obj.get_value("eta_val")
    # store the expected output values
    request.cls.expect_jac: np.ndarray = \
        _my_out_obj.get_value(
            "my_jac"
            ).reshape((4, 4))
    request.cls.expect_hyp: bool = \
        _my_out_obj.get_value("my_hyp")
    request.cls.expect_jac_three: np.ndarray = \
        _my_out_obj.get_value(
            "my_jac_three"
            ).reshape((3,3))
    request.cls.expect_hyp_three: bool = \
        _my_out_obj.get_value("my_hyp_three")
        

@pytest.mark.usefixtures('setup_test_class_no_compute')
class TestHyperbolicityCheckNoCompute:
    """Tests the hyperbolicity-checking functionality for the no-compute case."""
    
    # attribute type declarations
    gammas: np.ndarray
    gammas_three: np.ndarray
    omegas: np.ndarray
    omegas_three: np.ndarray
    q_val: np.float64
    eta_val: np.float64
    expect_jac: np.ndarray
    expect_hyp: bool
    expect_jac_three: np.ndarray
    expect_hyp_three: bool
    
    # test jacobian-computing function
    def test_jac_no_computer(self):
        """Tests the Jacobian matrix-computing functionality."""
        # compute the Jacobian matrix
        _my_jac = hyper_jac(
            gammas=self.gammas,
            omegas=self.omegas,
            q=self.q_val,
            eta=self.eta_val
            )
        # get mask for NaN values before matching
        _mm = ~(np.isnan(_my_jac)
                | np.isnan(self.expect_jac))
        # assert the non-NaN Jacobian values are OK
        assert np.allclose(
            _my_jac[_mm], self.expect_jac[_mm]
            )

    # test jacobian-computing function
    def test_jac_no_computer_three(self):
        """Tests the Jacobian matrix-computing functionality."""
        # compute the Jacobian matrix
        _my_jac = hyper_jac_three(
            gammas=self.gammas_three,
            omegas=self.omegas_three,
            q=self.q_val,
            eta=self.eta_val
            )
        # get mask for NaN values before matching
        _mm = ~(np.isnan(_my_jac)
                | np.isnan(self.expect_jac_three))
        # assert the non-NaN Jacobian values are OK
        assert np.allclose(
            _my_jac[_mm], self.expect_jac_three[_mm]
            )

    # test whether the correct hyperbolicity
    # check result is obtained
    def test_check_result(self):
        """Tests the hyperbolicity-checking functionality."""
        # compute the Jacobian matrix
        _my_jac = hyper_jac(
            gammas=self.gammas,
            omegas=self.omegas,
            q=self.q_val,
            eta=self.eta_val
            )
        # obtain whether it is hyperbolic
        _my_hyp = check_hyper(
            my_jac=_my_jac
            )
        # assert the hyperbolicity value is OK
        assert _my_hyp == self.expect_hyp  
        
    # test whether the correct hyperbolicity
    # check result is obtained
    def test_check_result_three(self):
        """Tests the hyperbolicity-checking functionality."""
        # compute the Jacobian matrix
        _my_jac = hyper_jac_three(
            gammas=self.gammas_three,
            omegas=self.omegas_three,
            q=self.q_val,
            eta=self.eta_val
            )
        # obtain whether it is hyperbolic
        _my_hyp = check_hyper(
            my_jac=_my_jac
            )
        # assert the hyperbolicity value is OK
        assert _my_hyp == self.expect_hyp_three
