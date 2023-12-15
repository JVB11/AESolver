'''Pytest module used to test the specific check module for quadratic mode coupling.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
'''
# import statements
import pytest
import numpy as np
# import mock objects and setup object
from mock_frequency_handler_object \
    import MockFreqHandler
from mock_mode_info_object \
    import MockModeInfo as MmI
from setup_object import SetupObject
# import module to be tested
from aesolver.pre_checks import PreThreeQuad


# generate a yield fixture that initializes
# the setup object        
@pytest.fixture(scope='class')
def setup_object(request) -> SetupObject:
    yield SetupObject(**request.param) 


# generate a fixture for the mock frequency handler
@pytest.fixture(scope='class')
def freq_handler_object(request) -> MockFreqHandler:
    yield MockFreqHandler(**request.param)


# generate a fixture that sets up the test class
@pytest.fixture(
    scope='class',
    params=[
        {'lm': [(2, 2), (1, 1), (1, 1)],
         'fail': False, 'l': [2, 1, 1],
         'm': [2, 1, 1], 'k': [0, 0, 0],
         'reason': None},
        {'lm': [(2, 1), (1, 1), (1, 1)],
         'fail': True, 'l': [2, 1, 1],
         'm': [1, 1, 1], 'k': [1, 0, 0],
         'reason': 'Azimuthal selection rule fails.'}                
    ]
    )
def setup_test_class(request) -> None:
    """Fixture used to set up the data used in the test class."""
        # set up the gyre_ad parameter to be used
    # in the PreCheckQuadratic object
    request.cls.gyre_ad: list[MmI] = [
        MmI(*request.param['lm'][0]),
        MmI(*request.param['lm'][1]),
        MmI(*request.param['lm'][2]),
    ]
    # set up the fail parameter
    request.cls.fail: bool = request.param['fail']
    # set up the fail reason parameter
    request.cls.reason: str | None= request.param['reason']


# generate the test class
@pytest.mark.usefixtures('setup_test_class')
@pytest.mark.parametrize(
    'setup_object',
    [
        {'triads': None, 'conjugation': None,
         'all_fulfilled': True},
        {'triads': [(1, 2, 3)], 'conjugation': None,
         'all_fulfilled': True},
        {'triads': None, 'conjugation': [(True, False, False)],
         'all_fulfilled': True},
        {'triads': [(1, 2, 3)], 'conjugation': [(True, False, False)],
         'all_fulfilled': True},
    ],
    indirect=True
)
@pytest.mark.parametrize(
    'freq_handler_object',
    [
        {'driving_rates': np.ones(3, dtype=np.float64),
         'q_factor': np.array(1.0),
         'fail': True, 'reason': 'driven resonance',
         'parametric_check': 'This is a driven resonance.',
         'direct_check': 'This is a driven resonance.',
         'driven_check': None, 'two_mode_harmonic': False},
        {'driving_rates': np.array([-1., 1., 1.], dtype=np.float64),
         'q_factor': np.array(1.0),
         'fail': True, 'reason': 'direct three-mode resonance',
         'parametric_check': 'This is a direct resonance.',
         'direct_check': None, 'two_mode_harmonic': False,
         'driven_check': 'This is a direct resonance.'},
        {'driving_rates': np.array([1., -1., -1.], dtype=np.float64),
         'q_factor': np.array(1.0),
         'fail': True, 'reason': 'Stability criterion not fulfilled.',
         'parametric_check': None, 'two_mode_harmonic': False,
         'direct_check': 'This is a parametric resonance.',
         'driven_check': 'This is a parametric resonance.'},
        {'driving_rates': np.array([1., -2., -3.], dtype=np.float64),
         'q_factor': np.array(2.0),
         'fail': False, 'reason': None,
         'parametric_check': None, 'two_mode_harmonic': False,
         'direct_check': 'This is a parametric resonance.',
         'driven_check': 'This is a parametric resonance.'},
        {'driving_rates': np.ones(3, dtype=np.float64),
         'q_factor': np.array(1.0),
         'fail': True, 'reason': 'driven resonance',
         'parametric_check': 'This is a driven resonance.',
         'direct_check': 'This is a driven resonance.',
         'driven_check': None, 'two_mode_harmonic': True},
        {'driving_rates': np.array([-3., 1., 1.], dtype=np.float64),
         'q_factor': np.array(1.0),
         'fail': False, 'reason': None,
         'parametric_check': 'This is a direct resonance.',
         'direct_check': None, 'two_mode_harmonic': True,
         'driven_check': 'This is a direct resonance.'},
        {'driving_rates': np.array([-1., 1., 1.], dtype=np.float64),
         'q_factor': np.array(1.0),
         'fail': True, 'reason': 'Stability criterion not satisfied.',
         'parametric_check': 'This is a direct resonance.',
         'direct_check': None, 'two_mode_harmonic': True,
         'driven_check': 'This is a direct resonance.'},
        {'driving_rates': np.array([1., -2., -2.], dtype=np.float64),
         'q_factor': np.array(1.0),
         'fail': False, 'reason': None,
         'parametric_check': None, 'two_mode_harmonic': True,
         'direct_check': 'This is a parametric resonance.',
         'driven_check': 'This is a parametric resonance.'},
        {'driving_rates': np.array([1., -1., -1.], dtype=np.float64),
         'q_factor': np.array(1.0),
         'fail': True, 'reason': 'Stability criterion not satisfied.',
         'parametric_check': None, 'two_mode_harmonic': True,
         'direct_check': 'This is a parametric resonance.',
         'driven_check': 'This is a parametric resonance.'}      
    ],
    indirect=True
)
class TestThreeModeCheckObject:
    """Tests the PreThreeQuad object."""
    
    # attribute type declarations
    gyre_ad: list[MmI]
    fail: bool
    reason: str | None
    
    def test_initialization(self, setup_object: SetupObject,
                            freq_handler_object: MockFreqHandler) -> None:
        """Tests the initialization of the
        PreThreeQuad object.
        """
        # initialize the object
        PreThreeQuad(
            gyre_ad=self.gyre_ad, freq_handler=freq_handler_object,
            triads=setup_object.triads,
            conjugation=setup_object.conjugation,
            two_mode_harmonic=freq_handler_object.two_mode_harmonic
        )

    def test_parametric_check(self, setup_object: SetupObject,
                              freq_handler_object: MockFreqHandler) -> None:
        # initialize the object
        my_obj = PreThreeQuad(
            gyre_ad=self.gyre_ad, freq_handler=freq_handler_object,
            triads=setup_object.triads,
            conjugation=setup_object.conjugation,
            two_mode_harmonic=freq_handler_object.two_mode_harmonic
        )
        # obtain the pytest parameter
        my_param = freq_handler_object.return_parametric_check_info()
        # check if this is an xfail
        if len(my_param.marks) == 1:
            # check if the test fails
            assert (not my_obj._check_parametric())
            pytest.xfail(my_param.marks[0].kwargs['reason'])
        else:
            assert my_obj._check_parametric()

    def test_direct_check(self, setup_object: SetupObject,
                          freq_handler_object: MockFreqHandler) -> None:
        # initialize the object
        my_obj = PreThreeQuad(
            gyre_ad=self.gyre_ad, freq_handler=freq_handler_object,
            triads=setup_object.triads,
            conjugation=setup_object.conjugation,
            two_mode_harmonic=freq_handler_object.two_mode_harmonic
        )
        # obtain the pytest parameter
        my_param = freq_handler_object.return_direct_check_info()
        # check if this is an xfail
        if len(my_param.marks) == 1:
            # check if the test fails
            assert (not my_obj._check_direct())
            pytest.xfail(my_param.marks[0].kwargs['reason'])
        else:
            assert my_obj._check_direct()

    def test_driven_check(self, setup_object: SetupObject,
                          freq_handler_object: MockFreqHandler) -> None:
        # initialize the object
        my_obj = PreThreeQuad(
            gyre_ad=self.gyre_ad, freq_handler=freq_handler_object,
            triads=setup_object.triads,
            conjugation=setup_object.conjugation,
            two_mode_harmonic=freq_handler_object.two_mode_harmonic
        )
        # obtain the pytest parameter
        my_param = freq_handler_object.return_driven_check_info()
        # check if this is an xfail
        if len(my_param.marks) == 1:
            # check if the test fails
            assert (not my_obj._check_driven())
            pytest.xfail(my_param.marks[0].kwargs['reason'])
        else:
            assert my_obj._check_driven()

    def test_stability_check(self, setup_object: SetupObject,
                             freq_handler_object: MockFreqHandler) -> None:
        # initialize the object
        my_obj = PreThreeQuad(
            gyre_ad=self.gyre_ad, freq_handler=freq_handler_object,
            triads=setup_object.triads,
            conjugation=setup_object.conjugation,
            two_mode_harmonic=freq_handler_object.two_mode_harmonic
        )
        # perform the check and obtain the result
        my_check = my_obj.check()
        # get the azimuthal selection rule check result to
        # perform additional verifications
        az_check = my_obj.generic_check()
        # perform indexing if required for our tests
        if not setup_object.all_fulfilled:
            az_check = az_check[0]
        # now start asserting whether test gets correct result
        # - check if azimuthal selection rule is applied correctly
        if self.fail and self.reason:
            # assert it fails
            assert (not az_check) and (az_check == my_check)
            # pytest xfail
            pytest.xfail(self.reason)
        else:
            # check if the stability selection rules should be fulfilled
            # - obtain the pytest parameter
            my_param = freq_handler_object.return_param()
            # check if the parameter is marked as failing
            if len(my_param.marks) == 1:
                assert (not my_check) and az_check
                pytest.xfail(my_param.marks[0].kwargs['reason'])
            else:
                assert my_check and az_check
