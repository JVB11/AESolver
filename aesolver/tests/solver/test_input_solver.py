"""Python test module used to test input for the solver module of aesolver.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import modules
import pytest
import sys
from pathlib import Path
# import support modules

# import expected values

# import module to be tested
from aesolver.solver import QuadRotAESolver


# fixture for the test class
@pytest.fixture(scope='class')
def setup_test_class(request) -> None:
    # set up an input object
    request.cls.input_obj = None
    # add the path to the pytest file
    request.cls.pytest_file_path: Path = request.path


# define the test class
@pytest.mark.usefixtures('setup_test_class')
class TestQuadRotAESolver:
    """Tests the QuadRotAESolver object of AESolver."""

    # attribute type definitions
    pytest_file_path: Path

    # test the initialization of the object
    def test_initialization(self):
        """Tests initialization of the QuadRotAESolver object."""
        # initialize the Solver object
        my_obj = QuadRotAESolver(
            sys_args=sys.argv,
            rot_pct=20,
            use_complex=False,
            adiabatic=True,
            use_rotating_formalism=True,
            base_dir=None,
            alternative_directory_path=None,
            alternative_file_name=None,
            use_symbolic_profiles=True,
            use_brunt_def=True,
            mode_selection_dict=None,
            recompute_for_profile=False,
            nr_omp_threads=4,
            use_parallel=False,
            ld_function='eddington',
            numerical_integration_method='trapz',
            stationary_solving=True,
            ad_path=None,
            nad_path=None,
            inlist_path=None,
            toml_use=False,
            get_debug_info=False,
            terms_cc_conj=None,
            polytrope_comp=False,
            analytic_polytrope=False,
            called_from_pytest=True,
            pytest_path=self.pytest_file_path
        )
        # assert the object is initialized
        assert my_obj

    # test the initialization of the object
    def test_initialization_polytrope(self):
        """Tests initialization of the QuadRotAESolver object for polytropes."""
        # initialize the Solver object
        my_obj = QuadRotAESolver(
            sys_args=sys.argv,
            rot_pct=20,
            use_complex=False,
            adiabatic=True,
            use_rotating_formalism=True,
            base_dir=None,
            alternative_directory_path=None,
            alternative_file_name=None,
            use_symbolic_profiles=True,
            use_brunt_def=True,
            mode_selection_dict=None,
            recompute_for_profile=False,
            nr_omp_threads=4,
            use_parallel=False,
            ld_function='eddington',
            numerical_integration_method='trapz',
            stationary_solving=True,
            ad_path=None,
            nad_path=None,
            inlist_path=None,
            toml_use=False,
            get_debug_info=False,
            terms_cc_conj=None,
            polytrope_comp=True,
            analytic_polytrope=False,
            called_from_pytest=True,
            pytest_path=self.pytest_file_path
        )
        # assert the object is initialized
        assert my_obj
        # initialize the Solver object using analytic polytrope expressions
        my_obj_ana = QuadRotAESolver(
            sys_args=sys.argv,
            rot_pct=20,
            use_complex=False,
            adiabatic=True,
            use_rotating_formalism=True,
            base_dir=None,
            alternative_directory_path=None,
            alternative_file_name=None,
            use_symbolic_profiles=True,
            use_brunt_def=True,
            mode_selection_dict=None,
            recompute_for_profile=False,
            nr_omp_threads=4,
            use_parallel=False,
            ld_function='eddington',
            numerical_integration_method='trapz',
            stationary_solving=True,
            ad_path=None,
            nad_path=None,
            inlist_path=None,
            toml_use=False,
            get_debug_info=False,
            terms_cc_conj=None,
            polytrope_comp=True,
            analytic_polytrope=True,
            called_from_pytest=True,
            pytest_path=self.pytest_file_path
        )
        # assert the object is initialized
        assert my_obj_ana

    # test the initialization of the object
    def test_initialization_toml(self):
        """Tests initialization of the QuadRotAESolver object using a toml inlist. """
        # initialize the Solver object
        my_obj = QuadRotAESolver(
            sys_args=sys.argv,
            rot_pct=20,
            use_complex=False,
            adiabatic=True,
            use_rotating_formalism=True,
            base_dir=None,
            alternative_directory_path=None,
            alternative_file_name=None,
            use_symbolic_profiles=True,
            use_brunt_def=True,
            mode_selection_dict=None,
            recompute_for_profile=False,
            nr_omp_threads=4,
            use_parallel=False,
            ld_function='eddington',
            numerical_integration_method='trapz',
            stationary_solving=True,
            ad_path=None,
            nad_path=None,
            inlist_path=None,
            toml_use=True,
            get_debug_info=False,
            terms_cc_conj=None,
            polytrope_comp=False,
            analytic_polytrope=False,
            called_from_pytest=True,
            pytest_path=self.pytest_file_path
        )
        # assert the object is initialized
        assert my_obj

    # test the initialization of the object
    def test_initialization_polytrope_toml(self):
        """Tests initialization of the QuadRotAESolver object for polytropes using a toml inlist."""
        # initialize the Solver object
        my_obj = QuadRotAESolver(
            sys_args=sys.argv,
            rot_pct=20,
            use_complex=False,
            adiabatic=True,
            use_rotating_formalism=True,
            base_dir=None,
            alternative_directory_path=None,
            alternative_file_name=None,
            use_symbolic_profiles=True,
            use_brunt_def=True,
            mode_selection_dict=None,
            recompute_for_profile=False,
            nr_omp_threads=4,
            use_parallel=False,
            ld_function='eddington',
            numerical_integration_method='trapz',
            stationary_solving=True,
            ad_path=None,
            nad_path=None,
            inlist_path=None,
            toml_use=True,
            get_debug_info=False,
            terms_cc_conj=None,
            polytrope_comp=True,
            analytic_polytrope=False,
            called_from_pytest=True,
            pytest_path=self.pytest_file_path
        )
        # assert the object is initialized
        assert my_obj
        # initialize the Solver object using analytic polytrope expressions
        my_obj_ana = QuadRotAESolver(
            sys_args=sys.argv,
            rot_pct=20,
            use_complex=False,
            adiabatic=True,
            use_rotating_formalism=True,
            base_dir=None,
            alternative_directory_path=None,
            alternative_file_name=None,
            use_symbolic_profiles=True,
            use_brunt_def=True,
            mode_selection_dict=None,
            recompute_for_profile=False,
            nr_omp_threads=4,
            use_parallel=False,
            ld_function='eddington',
            numerical_integration_method='trapz',
            stationary_solving=True,
            ad_path=None,
            nad_path=None,
            inlist_path=None,
            toml_use=True,
            get_debug_info=False,
            terms_cc_conj=None,
            polytrope_comp=True,
            analytic_polytrope=True,
            called_from_pytest=True,
            pytest_path=self.pytest_file_path
        )
        # assert the object is initialized
        assert my_obj_ana

    # test the initialization of the object
    def test_coupling_polytrope_toml(self):
        """Tests initialization of the QuadRotAESolver object for polytropes using a toml inlist."""
        # initialize the Solver object
        my_obj = QuadRotAESolver(
            sys_args=sys.argv,
            rot_pct=20,
            use_complex=False,
            adiabatic=True,
            use_rotating_formalism=True,
            base_dir=None,
            alternative_directory_path=None,
            alternative_file_name=None,
            use_symbolic_profiles=True,
            use_brunt_def=True,
            mode_selection_dict=None,
            recompute_for_profile=False,
            nr_omp_threads=4,
            use_parallel=False,
            ld_function='eddington',
            numerical_integration_method='trapz',
            stationary_solving=True,
            ad_path=None,
            nad_path=None,
            inlist_path=None,
            toml_use=True,
            get_debug_info=False,
            terms_cc_conj=None,
            polytrope_comp=True,
            analytic_polytrope=False,
            called_from_pytest=True,
            pytest_path=self.pytest_file_path
        )
        # try to compute modes
        my_obj.compute_coupling_one_triad()
