'''Pytest module that tests the superclass 'SuperCouplingCoefficient' of the aesolver.coupling_coefficients module

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
'''
# import statements
import pytest
import typing
# import support modules
from input_values_super_coupling_coefficient \
    import InputSuperCouplingCoefficient as ISCc
# import expected value modules
from expected_values_super_coupling_coefficient \
    import AttributesSuperCouplingCoefficientAfterInit \
        as ASCCai
# import module to be tested
from aesolver.coupling_coefficients.\
    coefficient_support_classes.\
        super_coupling_coefficients import \
            SuperCouplingCoefficient as SCc


# fixture for the test class
@pytest.fixture(scope='class')
def setup_test_class(request) -> None:
    # define the default input argument dictionary
    request.cls.default_input_dict = \
        ISCc.get_dict()
    # define the default output arguments/attributes
    request.cls.default_attr_dict = \
        ASCCai.get_dict()


# define the test class
@pytest.mark.usefixtures('setup_test_class')
class TestSuperCouplingCoefficient:
    """Tests the 'SuperCouplingCoefficient' class."""
    
    # attribute type declarations
    default_input_dict: dict[str, typing.Any]
    default_attr_dict: dict[str, typing.Any]
    
    # test initialization
    def test_init_scc(self):
        """Tests initialization of the SuperCouplingCoefficient object."""
        # initialize the object
        my_obj = SCc(
            **self.default_input_dict
            )
        # assert that went fine
        assert my_obj
        
    # test attributes after initialization
    def test_attrs_after_init(self):
        """Tests the values of the attributes of the SuperCouplingCoefficient object directly after initialization."""
        # initialize the object
        my_obj = SCc(
            **self.default_input_dict
            )
        # assert the attributes are as expected
        assert my_obj._kwargs_diff_div == \
            self.default_attr_dict['_kwargs_diff_div']
        assert my_obj._kwargs_diff_terms == \
            self.default_attr_dict['_kwargs_diff_terms']        
        assert my_obj._diff_terms_method == \
            self.default_attr_dict['_diff_terms_method']
        assert my_obj._list_common == \
            self.default_attr_dict['_list_common']
        assert my_obj._use_polytropic == \
            self.default_attr_dict['_use_polytropic']
        assert my_obj._nr_modes == \
            self.default_attr_dict['_nr_modes']
        assert my_obj._norm_P == \
            self.default_attr_dict['_norm_P']
        assert my_obj._norm_rho == \
            self.default_attr_dict['_norm_rho']
        assert my_obj._cut_center == \
            self.default_attr_dict['_cut_center']
