'''Pytest module used to test the generic functionalities of the aesolver.stationary_solving module.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
'''
# import statements
import pytest
import typing
# import support modules
from mock_solver_object_generic import \
    MockSolverGeneric as mSG
from mock_solver_object_generic import \
    ExpectedAttributes as EA
# import the module to be tested
from aesolver.stationary_solving.generic \
    import StatSolve    
    

# fixture for the test class
@pytest.fixture(scope='class')
def setup_test_class(request) -> None:
    # store the nr of modes used
    # in the coupling
    request.cls.nr_modes: int = 3
    # store the mock solver object
    # for three-mode coupling
    request.cls.mock_solver: mSG = mSG(
        nr_modes=request.cls.nr_modes
        )
    # store the expected attribute
    # dictionary
    request.cls.expect_dict: dict[str, typing.Any] = \
        EA.get_dict()
    

# define the test class
@pytest.mark.usefixtures('setup_test_class')
class TestGenericStatSolve:
    """Tests the generic functionalities of the aesolver.stationary_solve module."""
    
    # attribute type declarations
    nr_modes: int
    mock_solver: mSG
    expect_dict: dict[str, typing.Any]
    
    def test_initialization(self) -> None:
        """Tests the initialization of the generic stationary solving object."""
        # initialize the generic stationary solving object
        my_obj = StatSolve(
            solver_object=self.mock_solver,
            nr_modes=self.nr_modes
            )
        # check if the object was made
        assert my_obj
        # check if the object contains the expected attributes
        assert my_obj._coupler == self.expect_dict['_coupler']
        assert my_obj._nr_modes == self.expect_dict['_nr_modes']
        assert my_obj._mode_range == self.expect_dict['_mode_range']
        assert (my_obj._surf_lums == self.expect_dict['_surf_lums']).all()
        assert (my_obj._surf_xirs == self.expect_dict['_surf_xirs']).all()
        assert (my_obj._tk == self.expect_dict['_tk']).all()
        assert (my_obj._ek == self.expect_dict['_ek']).all()
        for _i in range(1, self.nr_modes + 1):
            assert getattr(my_obj, f'_a{_i}') == \
                self.expect_dict[f'_a{_i}']
            assert getattr(my_obj, f'_f{_i}') == \
                self.expect_dict[f'_f{_i}']
    
    def test_disc_multiplication_factors(self) -> None:
        """Tests the computation of the disc integral multiplication factors."""
        # initialize the generic stationary solving object
        my_obj = StatSolve(
            solver_object=self.mock_solver,
            nr_modes=self.nr_modes
            )
        # compute the disc integral multiplication factors
        my_obj._compute_disc_integral_multiplication_factors()
        # assert that the result is ok
        assert (my_obj._disc_multi_factors == \
            self.expect_dict['_disc_multi_factors']).all()      
    
    def test_create_array(self) -> None:
        """Tests the array creation method."""
        # initialize the generic stationary solving object
        my_obj = StatSolve(
            solver_object=self.mock_solver,
            nr_modes=self.nr_modes
            )
        # use the array creation method
        a_arr = my_obj._create_array_from_attrs('_a')
        f_arr = my_obj._create_array_from_attrs('_f')
        # assert that the result are ok
        assert (a_arr == self.expect_dict['_a_arr']).all()
        assert (f_arr == self.expect_dict['_f_arr']).all()
        