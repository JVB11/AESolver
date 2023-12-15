'''Pytest module used to test the three-mode functionalities of the aesolver.stationary_solving module.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
'''
# import statements
import pytest
import typing
import numpy as np
# import support module
from mock_setup_object import MockSetupObjectDefault as MSOd
from mock_setup_object import MockSetupObject as MSO
from mock_setup_object import MockSetupObjectComplexDisc as MSOcD
from compare_dicts import compare_dicts
# import the expected output enumeration objects
from expected_output_three_mode_statsolver \
    import DefaultOutputPreCheckNotOk as DoPCNo
from expected_output_three_mode_statsolver \
    import DefaultOutputPreCheckOk as DoPCo
from expected_output_three_mode_statsolver \
    import DefaultOutputPreCheckOkDisc as DoPCoD
from expected_output_three_mode_statsolver \
    import DefaultOutputPreCheckOkDiscComplex as DoPCoDc
# import the module to be tested
from aesolver.stationary_solving.three_mode \
    import ThreeModeStationary as TMS


# yield fixture for the mock setup object containing customized
# values for the disc integral factors
@pytest.fixture(scope='class')
def setup_object_disc(request) -> MSO:
    yield MSO(**request.param)


# yield fixture for the mock setup object containing customized
# complex values for the disc integral factors
@pytest.fixture(scope='class')
def setup_object_disc_complex(request) -> MSOcD:
    yield MSOcD(**request.param)


# yield fixture for the mock setup object
@pytest.fixture(scope='class')
def setup_object(request) -> MSOd:
    yield MSOd(**request.param)


# fixture for the test class
@pytest.fixture(scope='class')
def setup_test_class(request) -> None:
    # store variables used to initialize the ThreeModeStationary object
    request.cls.spin_incl: float | None = None
    request.cls.periastron_arg: float | None = None
    # store the nr of modes used in the coupling
    request.cls.nr_modes: int = 3


# define the test parameter list, which contains the parameters
# over which should be looped
parameter_list = [
    {'mode_frequencies': np.array([2.0, 3.0, 4.0]), 
         'driving_rates': np.array([1.0, -1.0, -1.0])},
    {'mode_frequencies': np.array([2.0, 3.0, 4.0]), 
        'driving_rates': np.array([1.0, -1.0, -1.0]),
        'azimuthal_wavenumbers': None},
    {'mode_frequencies': np.array([2.0, 3.0, 4.0]), 
        'driving_rates': np.array([1.0, -1.0, -1.0]),
        'azimuthal_wavenumbers': [2, 2, 2]},
    {'mode_frequencies': np.array([2.0, 3.0, 4.0]), 
        'driving_rates': np.array([1.0, -1.0, -1.0]),
        'nr_vals': 5},
    {'mode_frequencies': np.array([2.0, 3.0, 4.0]), 
        'driving_rates': np.array([1.0, -1.0, -1.0]),
        'cc_val': 2.0},
    {'mode_frequencies': np.array([2.0, 3.0, 4.0]), 
        'driving_rates': np.array([1.0, -1.0, -1.0]),
        'hr_values': None},
    {'mode_frequencies': np.array([2.0, 3.0, 4.0]), 
        'driving_rates': np.array([1.0, -1.0, -1.0]),
        'hr_values': [
            np.ones((5,), dtype=np.float64),
            np.ones((5,), dtype=np.float64),
            np.ones((5,), dtype=np.float64),
            ]},
    {'mode_frequencies': np.array([2.0, 3.0, 4.0]), 
        'driving_rates': np.array([1.0, -1.0, -1.0]),
        'azimuthal_wavenumbers': None, 'cc_val': 2.0,
        'hr_values': None, 'nr_vals': 5},
]


# define the test class for default values
@pytest.mark.usefixtures('setup_test_class')
@pytest.mark.parametrize(
    'setup_object', parameter_list, indirect=True
)
class TestThreeModeStatSolverDefault:
    """Tests the 'ThreeModeStationary' object using default input values for the disc factors."""
    
    # attribute type declarations
    nr_modes: int
    spin_incl: float | None
    periastron_arg: float | None
    
    # tests the initialization of the object
    def test_init_obj_precheck_ok(
        self, setup_object: MSOd
        ) -> None:
        """Tests the initialization method of the object.

        Notes
        -----
        Handles the case where the pre-check was satisfied.
        """
        # initialize the object
        my_obj = TMS(
            precheck_satisfied=True,
            freq_handler=setup_object.freq_handler,
            hyperbolic=True,
            solver_object=setup_object.solver,
            nr_modes=self.nr_modes,
            spin_inclination=self.spin_incl,
            periastron_arg=self.periastron_arg
            )
        # assert it was initialized
        assert my_obj

    # tests the initialization of the object
    def test_init_obj_precheck_false(
        self, setup_object: MSOd
        ) -> None:
        """Tests the initialization method of the object.

        Notes
        -----
        Handles the case where the pre-check was NOT satisfied.
        """
        # initialize the object
        my_obj = TMS(
            precheck_satisfied=False,
            freq_handler=setup_object.freq_handler,
            hyperbolic=True,
            solver_object=setup_object.solver,
            nr_modes=self.nr_modes,
            spin_inclination=self.spin_incl,
            periastron_arg=self.periastron_arg
            )
        # assert it was initialized
        assert my_obj
        
    def test_init_parameters_precheck_ok_solver(
        self, setup_object: MSOd
        ) -> None:
        """Tests the initialization parameters of the object.
        
        Notes
        -----
        Handles the case where the pre-check was satisfied, and a solver object was passed.
        """
        # initialize the object
        my_obj = TMS(
            precheck_satisfied=True,
            freq_handler=setup_object.freq_handler,
            hyperbolic=True,
            solver_object=setup_object.solver,
            nr_modes=self.nr_modes,
            spin_inclination=self.spin_incl,
            periastron_arg=self.periastron_arg
            )
        # assert whether the input parameters have the correct values
        assert my_obj._pre == True 
        assert my_obj._freq_handler == setup_object.freq_handler
        assert my_obj._coupler == setup_object.solver._coupler
        assert (my_obj._qf_products == setup_object.freq_handler.quality_factor_products).all()
        assert my_obj._cc_product == 4.0  # cc_val = 2.0
        assert (my_obj._lin_drive_rates == setup_object.freq_handler.driving_rates).all()
        assert my_obj._gamma == setup_object.freq_handler.gamma_sum
        assert my_obj._detuning == -5.0  # freqs = [2.0,3.0,4.0]
        assert my_obj._q == 5.0  # driving rates = [1.0,-1.0,-1.0]
        assert my_obj._q_crit == 2.5  # driving rates[1:] used
        assert my_obj._mu_val_spin_inclination == self.spin_incl
        assert my_obj._az_ang_periastron == self.periastron_arg
        assert my_obj._hyperbolic  # should be True

    def test_init_parameters_precheck_ok_no_solver(
        self, setup_object: MSOd
        ) -> None:
        """Tests the initialization parameters of the object.
        
        Notes
        -----
        Handles the case where the pre-check was satisfied, and a solver object was NOT passed.
        """
        # initialize the object
        my_obj = TMS(
            precheck_satisfied=True,
            freq_handler=setup_object.freq_handler,
            hyperbolic=True,
            solver_object=None,
            nr_modes=self.nr_modes,
            spin_inclination=self.spin_incl,
            periastron_arg=self.periastron_arg
            )
        # assert whether the input parameters have the correct values
        assert my_obj._pre == True 
        assert my_obj._freq_handler == setup_object.freq_handler
        assert my_obj._coupler == None
        assert (my_obj._lin_drive_rates == setup_object.freq_handler.driving_rates).all()
        assert my_obj._gamma == setup_object.freq_handler.gamma_sum
        assert my_obj._detuning == -5.0  # freqs = [2.0,3.0,4.0]
        assert my_obj._q == 5.0  # driving rates = [1.0,-1.0,-1.0]
        assert my_obj._q_crit == 2.5  # driving rates[1:] used
        assert my_obj._mu_val_spin_inclination == self.spin_incl
        assert my_obj._az_ang_periastron == self.periastron_arg
        assert my_obj._hyperbolic  # should be True
        
    def test_init_parameters_precheck_false_solver(
        self, setup_object: MSOd
        ) -> None:
        """Tests the initialization parameters of the object.
        
        Notes
        -----
        Handles the case where the pre-check was NOT satisfied, and a solver object was passed.
        """
        # initialize the object
        my_obj = TMS(
            precheck_satisfied=False,
            freq_handler=setup_object.freq_handler,
            hyperbolic=True,
            solver_object=setup_object.solver,
            nr_modes=self.nr_modes,
            spin_inclination=self.spin_incl,
            periastron_arg=self.periastron_arg
            )
        # assert whether the input parameters have the correct values
        assert my_obj._pre == False 
        assert my_obj._freq_handler == setup_object.freq_handler
        assert my_obj._coupler == setup_object.solver._coupler
        assert (my_obj._lin_drive_rates == setup_object.freq_handler.driving_rates).all()
        assert my_obj._gamma == setup_object.freq_handler.gamma_sum
        assert my_obj._detuning == -5.0  # freqs = [2.0,3.0,4.0]
        assert my_obj._q == 5.0  # driving rates = [1.0,-1.0,-1.0]
        assert my_obj._q_crit == 2.5  # driving rates[1:] used
        assert my_obj._mu_val_spin_inclination == self.spin_incl
        assert my_obj._az_ang_periastron == self.periastron_arg
        assert my_obj._hyperbolic  # should be True

    def test_init_parameters_precheck_false_no_solver(
        self, setup_object: MSOd
        ) -> None:
        """Tests the initialization parameters of the object.
        
        Notes
        -----
        Handles the case where the pre-check was NOT satisfied, and a solver object was NOT passed.
        """
        # initialize the object
        my_obj = TMS(
            precheck_satisfied=False,
            freq_handler=setup_object.freq_handler,
            hyperbolic=True,
            solver_object=None,
            nr_modes=self.nr_modes,
            spin_inclination=self.spin_incl,
            periastron_arg=self.periastron_arg
            )
        # assert whether the input parameters have the correct values
        assert my_obj._pre == False 
        assert my_obj._freq_handler == setup_object.freq_handler
        assert my_obj._coupler == None
        assert (my_obj._lin_drive_rates == setup_object.freq_handler.driving_rates).all()
        assert my_obj._gamma == setup_object.freq_handler.gamma_sum
        assert my_obj._detuning == -5.0  # freqs = [2.0,3.0,4.0]
        assert my_obj._q == 5.0  # driving rates = [1.0,-1.0,-1.0]
        assert my_obj._q_crit == 2.5  # driving rates[1:] used
        assert my_obj._mu_val_spin_inclination == self.spin_incl
        assert my_obj._az_ang_periastron == self.periastron_arg
        assert my_obj._hyperbolic  # should be True

    def test_call_precheck_false(
        self, setup_object: MSOd
        ) -> None:
        # initialize the object
        my_obj = TMS(
            precheck_satisfied=False,
            freq_handler=setup_object.freq_handler,
            hyperbolic=True,
            solver_object=setup_object.solver,
            nr_modes=self.nr_modes,
            spin_inclination=self.spin_incl,
            periastron_arg=self.periastron_arg
            )
        # receive the output dictionary
        output_dict: dict[str, typing.Any] = my_obj()
        # assert it confirms to expectations
        assert compare_dicts(
            first=output_dict,
            second=DoPCNo.get_replaced_dict()
            )

    def test_call_precheck_ok(
        self, setup_object: MSOd
        ) -> None:
        # initialize the object
        my_obj = TMS(
            precheck_satisfied=True,
            freq_handler=setup_object.freq_handler,
            hyperbolic=True,
            solver_object=setup_object.solver,
            nr_modes=self.nr_modes,
            spin_inclination=self.spin_incl,
            periastron_arg=self.periastron_arg
            )
        # receive the output dictionary
        output_dict: dict[str, typing.Any] = my_obj()
        # assert it confirms to expectations
        assert compare_dicts(
            first=output_dict,
            second=DoPCo.get_replaced_dict()
            )
        # assert that the extra attributes are of OK value
        for _i in range(1, 4):
            assert getattr(my_obj, f"_f{_i}") == \
                np.zeros((1,), dtype=np.float64)[0]
        assert my_obj._rel_lum_phase == \
            np.zeros((1,), dtype=np.float64)[0]
        assert np.allclose(my_obj._houghs_observer,
                           np.ones((3,), dtype=np.float64))
        assert np.allclose(my_obj._az_factors,
                           np.zeros((3,), dtype=np.float64))
        assert my_obj._az_ang_periastron == None
        assert my_obj._mu_val_spin_inclination == None

# define the test class for non-default values
@pytest.mark.usefixtures('setup_test_class')
@pytest.mark.parametrize(
    'setup_object_disc', parameter_list, indirect=True
)
class TestThreeModeStatSolverDisc:
    """Tests the 'ThreeModeStationary' object using customized input values for the disc factors."""
    
    def test_call_precheck_ok_adjusted_disc_multi_factors(
        self, setup_object_disc: MSO
        ) -> None:
        # initialize the object
        my_obj = TMS(
            precheck_satisfied=True,
            freq_handler=setup_object_disc.freq_handler,
            hyperbolic=True,
            solver_object=setup_object_disc.solver,
            nr_modes=self.nr_modes,
            spin_inclination=self.spin_incl,
            periastron_arg=self.periastron_arg
            )
        # receive the output dictionary
        output_dict: dict[str, typing.Any] = my_obj()
        # assert it confirms to expectations
        assert compare_dicts(
            first=output_dict,
            second=DoPCoD.get_replaced_dict()
            )
        # assert that the extra attributes are of OK value
        for _i in range(1, 4):
            assert getattr(my_obj, f"_f{_i}") == \
                getattr(my_obj, f"_a{_i}")
        assert my_obj._rel_lum_phase == \
            np.zeros((1,), dtype=np.float64)[0]
        assert np.allclose(my_obj._houghs_observer,
                           np.ones((3,), dtype=np.float64))
        assert np.allclose(my_obj._az_factors,
                           np.zeros((3,), dtype=np.float64))
        assert my_obj._az_ang_periastron == None
        assert my_obj._mu_val_spin_inclination == None


# define the test class for non-default complex values
@pytest.mark.usefixtures('setup_test_class')
@pytest.mark.parametrize(
    'setup_object_disc_complex', parameter_list, indirect=True
)
class TestThreeModeStatSolverDiscComplex:
    """Tests the 'ThreeModeStationary' object using customized input values for the complex disc factors."""
    
    def test_call_precheck_ok_adjusted_disc_multi_factors(
        self, setup_object_disc_complex: MSOcD
        ) -> None:
        # initialize the object
        my_obj = TMS(
            precheck_satisfied=True,
            freq_handler=setup_object_disc_complex.freq_handler,
            hyperbolic=True,
            solver_object=setup_object_disc_complex.solver,
            nr_modes=self.nr_modes,
            spin_inclination=self.spin_incl,
            periastron_arg=self.periastron_arg
            )
        # receive the output dictionary
        output_dict: dict[str, typing.Any] = my_obj()
        # assert it confirms to expectations
        assert compare_dicts(
            first=output_dict,
            second=DoPCoDc.get_replaced_dict()
            )
        # assert that the extra attributes are of OK value
        for _i in range(1, 4):
            assert getattr(my_obj, f"_f{_i}") == \
                np.sqrt(2) * getattr(my_obj, f"_a{_i}")
        assert my_obj._rel_lum_phase == \
            np.zeros((1,), dtype=np.float64)[0]
        assert np.allclose(my_obj._houghs_observer,
                           np.ones((3,), dtype=np.float64))
        assert np.allclose(my_obj._az_factors,
                           np.zeros((3,), dtype=np.float64))
        assert my_obj._az_ang_periastron == None
        assert my_obj._mu_val_spin_inclination == None
