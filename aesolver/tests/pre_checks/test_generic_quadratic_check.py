'''Pytest module used to perform the generic quadratic check for the aesolver.pre_checks module.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
'''
# import statements
import pytest
# import class to be tested
from aesolver.pre_checks import PreCheckQuadratic
# local import of mock class and setup class
from mock_mode_info_object import \
    MockModeInfo as MmI
from setup_object import SetupObject


# generate a yield fixture that sets up an object
# that contains the 'triads', 'conjugation'
# and 'all_fulfilled' setup    
@pytest.fixture(scope='class')
def setup_object(request) -> SetupObject:
    yield SetupObject(**request.param)    


# generate a fixture that sets up gyre ad
# for the test class
@pytest.fixture(
    scope='class', 
    params=[
        {'lm': [(2, 2), (1, 1), (1, 1)],
         'fail': False, 'l': [2, 1, 1],
         'm': [2, 1, 1], 'k': [0, 0, 0]},
        {'lm': [(2, 1), (1, 1), (1, 1)],
         'fail': True, 'l': [2, 1, 1],
         'm': [1, 1, 1], 'k': [1, 0, 0]},
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
    # set up the expected list output parameters
    request.cls.expected_l: list[int] = request.param['l']
    request.cls.expected_m: list[int] = request.param['m']
    request.cls.expected_k: list[int] = request.param['k']


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
        {'triads': None, 'conjugation': None,
         'all_fulfilled': False},
        {'triads': [(1, 2, 3)], 'conjugation': None,
         'all_fulfilled': False},
        {'triads': None, 'conjugation': [(True, False, False)],
         'all_fulfilled': False},
        {'triads': [(1, 2, 3)], 'conjugation': [(True, False, False)],
         'all_fulfilled': False},
    ],
    indirect=True
)
class TestPreCheckQuadratic:
    """Tests the PreCheckQuadratic object.
    """
    # attribute type declarations
    gyre_ad: list[MmI]
    fail: bool
    expected_l: list[int]
    expected_m: list[int]
    expected_k: list[int]
    
    def test_initialization(self, setup_object: SetupObject) -> None:
        """Tests the initialization of the PreCheckQuadratic object."""
        # initialize the object
        PreCheckQuadratic(
            gyre_ad=self.gyre_ad,
            triads=setup_object.triads,
            conjugation=setup_object.conjugation,
            all_fulfilled=setup_object.all_fulfilled
        )

    def test_lists(self, setup_object: SetupObject) -> None:
        """Tests if the correct lists are available."""
        # initialize the object
        my_obj = PreCheckQuadratic(
            gyre_ad=self.gyre_ad,
            triads=setup_object.triads,
            conjugation=setup_object.conjugation,
            all_fulfilled=setup_object.all_fulfilled
        )
        # check if the lists are ok
        assert all([
            my_obj.l_list[0] == self.expected_l,
            my_obj.m_list[0] == self.expected_m,
            my_obj.k_list[0] == self.expected_k,
        ])

    def test_defaults(self, setup_object: SetupObject) -> None:
        """Tests methods using the default values for the generic_check method of the PreCheckQuadratic object."""
        # initialize the object
        my_obj = PreCheckQuadratic(
            gyre_ad=self.gyre_ad,
            triads=setup_object.triads,
            conjugation=setup_object.conjugation,
            all_fulfilled=setup_object.all_fulfilled
        )
        # check if the generic check results
        # are as expected
        if setup_object.all_fulfilled:
            assert (not self.fail) == my_obj.generic_check()
        else:
            assert (not self.fail) == my_obj.generic_check()[0]   
