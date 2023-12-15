'''Pytest module that tests the class 'QuadraticCouplingCoefficientRotating' of the aesolver.coupling_coefficients module

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
'''
# import statements
import math
import pytest
import typing
import numpy as np
from collections import defaultdict
# import support modules
from mock_gyre_detail import GYREDetailDictData as GDdd
from input_values_quadratic_coupling_coefficient_rotating import \
    InputQuadraticCouplingCoefficientRotating as iQCCR
from aesolver.frequency_handling import FH
from pytest_util_functions import value_comparison,\
    compare_elements, compare_dicts
# import expected value modules
from expected_values_quadratic_coupling_coefficient_rotating import \
    ExpectedCommonAttrsAfterInitQCCR as eCAaiQCCR
from expected_values_quadratic_coupling_coefficient_rotating import \
    ExpectedModeSpecificAttrsAfterInitQCCR as eMSAaiQCCR
from expected_values_quadratic_coupling_coefficient_rotating import \
    ExpectedProfileAttrsAfterInitQCCR as ePAaiQCCR
from expected_values_quadratic_coupling_coefficient_rotating import \
    ExpectedComputedAttrsAfterInitQCCR as eCompAaiQCCR
from expected_values_quadratic_coupling_coefficient_rotating import \
    ExpectedRadialVectorAttrsAfterInitQCCR as eRVAaiQCCR
from expected_values_quadratic_coupling_coefficient_rotating import \
    ExpectedRadialVectorGradientAttrsAfterInitQCCR as eRVGAaiQCCR
from expected_values_quadratic_coupling_coefficient_rotating import \
    ExpectedRadialDivergenceAttrAfterInitQCCR as eRDAaiQCCR
from expected_norm_factors import NormFactorValueMode1, \
    NormFactorValueMode2, NormFactorValueMode3
# import module to be tested
from aesolver.coupling_coefficients import QCCR


# define comparison function
def comparison(my_obj_val: typing.Any,
               expected_val: typing.Any,
               len_enforced: bool=True) -> bool:
    """Performs comparison of values.

    Parameters
    ----------
    my_obj_val : typing.Any
        Value from the object that is tested.
    val : typing.Any
        Expected value.
    len_enforced : bool, optional
        Assert that the lengths of the two input dictionaries are the same; by default True.

    Returns
    -------
    bool
        Whether the two input values are comparable.
    """
    if isinstance(my_obj_val, dict):
        return compare_dicts(
            first=my_obj_val, second=expected_val,
            len_enforced=len_enforced
            )
    elif isinstance(my_obj_val, list | np.ndarray):
        try:
            return compare_elements(
            first=my_obj_val,
            second=expected_val
            )
        except TypeError:
            return my_obj_val == expected_val
    else:
        return value_comparison(
            first=my_obj_val, second=expected_val
            ) 


# fixture for the test class
@pytest.fixture(scope='class')
def setup_test_class(request) -> None:
    # set up an input value object
    request.cls.input_value_object: \
        dict[str, typing.Any] = iQCCR.get_dict()
    # initialize a frequency handler object
    my_fh = FH(GDdd.get_dict(mode_nr=1, nad=False),
               GDdd.get_dict(mode_nr=2, nad=False),
               GDdd.get_dict(mode_nr=3, nad=False))
    # load the non-adiabatic frequencies
    my_fh.set_mode_nonadiabatic_info(
        GDdd.get_dict(mode_nr=1, nad=True),
        GDdd.get_dict(mode_nr=2, nad=True),
        GDdd.get_dict(mode_nr=3, nad=True)
        )
    # set up a frequency handler object
    request.cls.my_fh: FH = my_fh
    # set up the dictionary of common attributes after
    # initialization
    request.cls.common_attr_init: dict[str, typing.Any]= \
        eCAaiQCCR.get_dict()
    # set up the dictionary of mode-specific attributes
    # after initialization
    request.cls.mode_spec_attr_init: \
        dict[str, dict[str, typing.Any]]= \
            {
                1: eMSAaiQCCR.get_dict(mode_nr=1),
                2: eMSAaiQCCR.get_dict(mode_nr=2),
                3: eMSAaiQCCR.get_dict(mode_nr=3)
            }
    # set up the dictionary of (MESA) profile attributes
    # after initialization
    request.cls.profile_attr_init: dict[str, typing.Any]= \
        ePAaiQCCR.get_dict()
    # expected values for lists
    request.cls.expect_l_list: list[int] = \
        eMSAaiQCCR.l.tolist()
    request.cls.expect_m_list: list[int] = \
        eMSAaiQCCR.m.tolist()
    request.cls.expect_lambda_list: list[complex] = \
        eMSAaiQCCR.my_lambda
    # set up the dictionary of expected computed values
    request.cls.computed_value_dict: dict[str, typing.Any]= \
        eCompAaiQCCR.get_dict()
    # set up the dictionary of expected radial vector values
    request.cls.rad_vecs: dict[str, dict[str, typing.Any]]= \
        {
            1: eRVAaiQCCR.get_dict(mode_nr=1),
            2: eRVAaiQCCR.get_dict(mode_nr=2),
            3: eRVAaiQCCR.get_dict(mode_nr=3)            
        }
    # set up the dictionary of expected radial vector gradient values
    request.cls.rad_vec_grads: dict[str, dict[str, typing.Any]]= \
        {
            1: eRVGAaiQCCR.get_dict(mode_nr=1),
            2: eRVGAaiQCCR.get_dict(mode_nr=2),
            3: eRVGAaiQCCR.get_dict(mode_nr=3)
        }
    # set up the dictionary of expected radial divergence values
    request.cls.rad_divs: dict[str, dict[str, typing.Any]]= \
        {
            1: eRDAaiQCCR.get_dict(mode_nr=1),
            2: eRDAaiQCCR.get_dict(mode_nr=2),
            3: eRDAaiQCCR.get_dict(mode_nr=3)
        }
        

# define the test class
@pytest.mark.usefixtures('setup_test_class')
class TestQuadraticCouplingCoefficientRotating:
    """Tests the QuadraticCouplingCoefficientRotating class."""
    
    # attribute type declarations
    input_value_object: dict[str, typing.Any]
    my_fh: FH
    common_attr_init: dict[str, typing.Any]
    mode_spec_attr_init: dict[str, dict[str, typing.Any]]
    profile_attr_init: dict[str, typing.Any]
    expect_l_list: list[int]
    expect_m_list: list[int]
    expect_lambda_list: list[complex]
    computed_value_dict: dict[str, typing.Any]
    rad_vecs: dict[str, dict[str, typing.Any]]
    rad_vec_grads: dict[str, dict[str, typing.Any]]
    rad_divs: dict[str, dict[str, typing.Any]]
    
    # test initialization
    def test_init_qccr(self):
        """Tests the initialization of the QuadraticCouplingCoefficientRotating object."""
        # initialize the object
        my_obj = QCCR(
            **self.input_value_object,
            freq_handler=self.my_fh
            )
        # assert the initialization went ok
        my_obj.print_all_debug_properties()
        assert my_obj
        
    # test attributes after initialization
    def test_attrs_after_init(self):
        """Tests the attributes of the QuadraticCouplingCoefficientRotating object."""
        # initialize the object
        my_obj = QCCR(
            **self.input_value_object,
            freq_handler=self.my_fh
            )
        # assert that the attributes
        # after initialization are ok
        # - simple initializations
        assert my_obj._use_complex == \
            iQCCR.use_complex
        assert my_obj._conj_cc == \
            (True, False, False)
        assert my_obj._omp_threads == \
            iQCCR.nr_omp_threads
        assert my_obj._use_parallel == \
            iQCCR.use_parallel
        assert my_obj._numerical_integration_method == \
            iQCCR.numerical_integration_method
        assert my_obj._use_symbolic_derivative == \
            iQCCR.use_symbolic_derivatives
        assert my_obj._npts == iQCCR.npoints
        assert my_obj._self_coupling == \
            iQCCR.self_coupling
        assert np.allclose(
            my_obj.x, self.common_attr_init['x']
            )
        # - radial integration method
        assert my_obj._radint == \
            my_obj._compute_radial_integral_real
        # - common attributes
        for _k, _v in self.common_attr_init.items():
            # get the attribute value
            _v_a = getattr(my_obj, f"_{_k}")
            assert comparison(
                my_obj_val=_v_a, expected_val=_v,
                len_enforced=False
                )
        # - mode-specific attributes
        for _nr, _v in self.mode_spec_attr_init.items():
            for _k, _val in _v.items():
                my_obj_val = getattr(my_obj, f"_{_k}_mode_{_nr}")
                assert comparison(
                    my_obj_val=my_obj_val, expected_val=_val,
                    len_enforced=False
                    )
        # - profile attributes (MESA)
        for _k, _v in self.profile_attr_init.items():
            my_obj_val = getattr(my_obj, _k)
            assert comparison(
                my_obj_val=my_obj_val, expected_val=_v
                )
        # - normalization factors
        assert my_obj._norm_P == \
            self.common_attr_init['P'].max()
        assert my_obj._norm_rho == \
            self.common_attr_init['rho'].max()
        # - lists
        assert my_obj._l_list == \
            self.expect_l_list
        assert my_obj._m_list == \
            self.expect_m_list
        assert my_obj._nu_list == \
            self.my_fh.spin_factors.tolist()
        assert compare_elements(
            self.expect_lambda_list,
            my_obj._lambda_list
            )  # TODO: CHECK WHY THIS IS NOT A SIMPLE LIST OF COMPLEX FLOATS?
        # - coupling coefficients
        assert my_obj.coupling_coefficient \
            == None
        assert my_obj.coupling_coefficient_profile \
            == None

    # # test computed attributes after initialization
    # def test_computed_attrs_after_init(self):
    #     """Tests the computed attributes of the
    #     QuadraticCouplingCoefficientRotating object.
    #     """
    #     # initialize the object
    #     my_obj = QCCR(
    #         **self.input_value_object,
    #         freq_handler=self.my_fh
    #         )
    #     # assert that the computed attributes are OK
    #     # - MESA / GYRE attribute loads
    #     for _k, _v in self.computed_value_dict.items():
    #         my_obj_val = getattr(my_obj, _k)
    #         assert comparison(
    #             my_obj_val=my_obj_val, expected_val=_v
    #             )
    #     # - normalization factors
    #     assert math.isclose(
    #         my_obj.normalization_factors[0],
    #         NormFactorValueMode1.norm_factor
    #         )
    #     assert math.isclose(
    #         my_obj.normalization_factors[1],
    #         NormFactorValueMode2.norm_factor
    #         )
    #     assert math.isclose(
    #         my_obj.normalization_factors[2],
    #         NormFactorValueMode3.norm_factor
    #         )
    #     # - radial vectors
    #     for _nr, _v in self.rad_vecs.items():
    #         _norm_fac = my_obj.normalization_factors[_nr - 1]
    #         for _k, _val in _v.items():
    #             my_obj_val = getattr(my_obj, f"_{_k}_mode_{_nr}")
    #             assert comparison(
    #                 my_obj_val=my_obj_val,
    #                 expected_val=_val * _norm_fac,
    #                 len_enforced=False
    #                 )
    #     # - radial gradients
    #     for _nr, _v in self.rad_vec_grads.items():
    #         _norm_fac = my_obj.normalization_factors[_nr - 1]
    #         for _k, _val in _v.items():
    #             my_obj_val = getattr(my_obj, f"_{_k}_mode_{_nr}")
    #             assert comparison(
    #                 my_obj_val=my_obj_val,
    #                 expected_val=_val * _norm_fac,
    #                 len_enforced=False
    #                 )       
    #     # - radial divergence
    #     for _nr, _v in self.rad_divs.items():
    #         _norm_fac = my_obj.normalization_factors[_nr - 1]
    #         for _k, _val in _v.items():
    #             my_obj_val = getattr(my_obj, f"_{_k}_mode_{_nr}")
    #             assert comparison(
    #                 my_obj_val=my_obj_val,
    #                 expected_val=_val * _norm_fac,
    #                 len_enforced=False
    #                 )

    # # test attributes after initialization
    # def test_rad_int(self):
    #     """Tests the radial integral method 'radint' of the
    #     QuadraticCouplingCoefficientRotating object.
    #     """
    #     # initialize the object
    #     my_obj = QCCR(
    #         **self.input_value_object,
    #         freq_handler=self.my_fh
    #         )
    #     # use the rad_int method to compute a radial integral
    #     # of an easy integrand (multiplying factor 1.0)
    #     my_integral = my_obj._rad_int(
    #         radial_terms=['_y_1', '_y_1', '_y_1'],
    #         multiplying_term=1.0
    #         )
    #     # assert that the result is ok
    #     expected_result = (1.9798583984375+0j)
    #     assert math.isclose(my_integral.real, expected_result.real) \
    #         and math.isclose(my_integral.imag, expected_result.imag)
    #     # using a different multiplying factor
    #     my_integral_multiplier = my_obj._rad_int(
    #         radial_terms=['_y_1', '_y_1', '_y_1'],
    #         multiplying_term=2.0
    #         )
    #     assert math.isclose(my_integral_multiplier.real,
    #                         2.0 * expected_result.real) \
    #         and math.isclose(my_integral_multiplier.imag,
    #                          2.0 * expected_result.imag)  
    #     # using a different indexer
    #     expected_result_index = (1.859130859375+0j)
    #     my_integral_indexer = my_obj._rad_int(
    #         radial_terms=['_y_1', '_y_1', '_y_1'],
    #         multiplying_term=1.0, indexer=np.s_[1:]           
    #         )
    #     assert math.isclose(my_integral_indexer.real,
    #                         expected_result_index.real) \
    #         and math.isclose(my_integral_indexer.imag,
    #                          expected_result_index.imag)          

    # # test map hough to mode method
    # def test_map_hough_to_mode(self):
    #     """Tests the '_map_hough_to_mode' method of the
    #     QuadraticCouplingCoefficientRotating object.        
    #     """
    #     # initialize the object
    #     my_obj = QCCR(
    #         **self.input_value_object,
    #         freq_handler=self.my_fh
    #         )
    #     # use the mapping method and assert the result is ok
    #     my_mapping = my_obj._map_hough_to_mode(
    #         ['hr', 'ht', 'hp'], [1, 2, 3]
    #         )
    #     assert my_mapping == ['_hr_mode_1', '_ht_mode_2',
    #                           '_hp_mode_3']
    #     # use another mapping
    #     my_mapping2 = my_obj._map_hough_to_mode(
    #         ['ht', 'hp'], [1, 1]
    #         )
    #     assert my_mapping2 == ['_ht_mode_1', '_hp_mode_1']
        
    # # test map hough to mode method
    # @pytest.mark.xfail(reason='Input lists not of same length!')
    # def test_map_hough_to_mode_fail_length(self):
    #     """Tests the failure branch of the '_map_hough_to_mode'
    #     method of the QuadraticCouplingCoefficientRotating object.        
    #     """
    #     # initialize the object
    #     my_obj = QCCR(
    #         **self.input_value_object,
    #         freq_handler=self.my_fh
    #         )
    #     # use the mapping method and assert the result is ok
    #     assert my_obj._map_hough_to_mode(
    #         ['hr', 'ht', 'hp'], [1, 2]
    #         )  # lists not of the same length

    # # test symmetry mapping method
    # def test_symmetry_mapping(self):
    #     """Tests the '_symmetry_mapping' method of the
    #     QuadraticCouplingCoefficientRotating object.
    #     """
    #     # initialize the object
    #     my_obj = QCCR(
    #         **self.input_value_object,
    #         freq_handler=self.my_fh
    #         )
    #     # perform a mapping
    #     my_mapping = my_obj._symmetry_mapping(['hr', 'ht', 'hp'])
    #     expected = defaultdict(list)
    #     expected[('hr', 'ht', 'hp')] = [0]
    #     expected[('hr', 'hp', 'ht')] = [1]
    #     expected[('ht', 'hr', 'hp')] = [2]
    #     expected[('ht', 'hp', 'hr')] = [3]
    #     expected[('hp', 'hr', 'ht')] = [4]
    #     expected[('hp', 'ht', 'hr')] = [5]
    #     assert expected == my_mapping
    #     # perform a more symmetric mapping
    #     my_mapping2 = my_obj._symmetry_mapping(['hp', 'ht', 'ht'])
    #     expected2 = defaultdict(list)
    #     expected2[('hp', 'ht', 'ht')] = [0, 1]
    #     expected2[('ht', 'hp', 'ht')] = [2, 4]
    #     expected2[('ht', 'ht', 'hp')] = [3, 5]
    #     assert expected2 == my_mapping2
    #     # perform symmetric mapping
    #     my_mapping3 = my_obj._symmetry_mapping(['ht', 'ht', 'ht'])
    #     expected3 = defaultdict(list)
    #     expected3[('ht', 'ht', 'ht')] = [0, 1, 2, 3, 4, 5]
    #     assert expected3 == my_mapping3

    # # test overloaded method that determines symmetry of expressions and
    # # computes the result of the S operator defined in Lee(2012)
    # def test_overloaded_symmetry_determiner(self):
    #     """Tests the overloaded method that determines symmetry
    #     of expressions and computes the result of the S operator
    #     defined in Lee(2012); as defined in the
    #     QuadraticCouplingCoefficientRotating object.
    #     """
    #     # initialize the object
    #     my_obj = QCCR(
    #         **self.input_value_object,
    #         freq_handler=self.my_fh
    #         )
    #     # use the overloaded symmetry determiner method
    #     my_S_operator_output = my_obj._overloaded_symmetry_determiner(
    #         ['hr', 'hr', 'hr'], ['_y_1', '_y_1', '_y_1'],
    #         radial_multiplier=1.0, div_by_sin=[False],
    #         ang_factor=[1.0], rad_factor=[1.0], rad_indexer=np.s_[:]
    #         )
    #     assert my_S_operator_output == 0.0  # angular selection rule NOT satisfied because m_list is WRONG!
    #     # adjust m list values to make the angular selection rule to be satisfied
    #     my_input_value_object_to_be_changed = self.input_value_object.copy()
    #     my_input_value_object_to_be_changed['m_list'] = [2, 1, 1]
    #     # use it get the S operator
    #     my_new_obj = QCCR(**my_input_value_object_to_be_changed,
    #                       freq_handler=self.my_fh)
    #     my_new_S_operator_output = my_new_obj._overloaded_symmetry_determiner(
    #         ['hr', 'hr', 'hr'], ['_y_1', '_y_1', '_y_1'],
    #         radial_multiplier=1.0, div_by_sin=[False],
    #         ang_factor=[1.0], rad_factor=[1.0], rad_indexer=np.s_[:]            
    #         )
    #     my_expected_radial = 1.9798583984375
    #     my_expected_angular = 0.6885235707004026
    #     my_new_expected_output = (my_expected_radial * 6.0) * my_expected_angular
    #     # (6.0 * rad_factor) * ang_factor
    #     # assert the output is as expected
    #     assert math.isclose(my_new_expected_output,
    #                         my_new_S_operator_output)
    #     # compute a different integral
    #     my_new_S_operator_output2 = my_new_obj._overloaded_symmetry_determiner(
    #         ['hr', 'hr', 'hr'], ['_y_1', '_y_1', '_y_1'],
    #         radial_multiplier=5.0, div_by_sin=[False],
    #         ang_factor=[1.0], rad_factor=[1.0], rad_indexer=np.s_[:]            
    #         )  
    #     assert math.isclose(my_new_expected_output * 5.0,
    #                         my_new_S_operator_output2) 
    #     # compute another integral
    #     my_new_S_operator_output_sin = my_new_obj._overloaded_symmetry_determiner(
    #         ['hr', 'hr', 'hr'], ['_y_1', '_y_1', '_y_1'],
    #         radial_multiplier=1.0, div_by_sin=[True],
    #         ang_factor=[1.0], rad_factor=[1.0], rad_indexer=np.s_[:]            
    #         )
    #     my_angular_blowup = 43.83502514667403  # NOTE: THIS BLOWS UP BECAUSE OF THE DIVISION BY SIN, WHEN USING THETAS(?)! --> FACTOR 63.665249836084364 LARGER
    #     my_expected_output_sin = my_expected_radial * my_angular_blowup * 6.0
    #     # APPARENTLY NO ANGULAR BLOWUP HERE
    #     my_angular_no_blowup = 0.6809015764892271  # NOTE: THIS DOES NOT BLOW UP WHEN COMPUTING WITH MUS?
    #     my_expected_output_sin_no_blowup = my_expected_radial * my_angular_no_blowup * 6.0
    #     assert math.isclose(my_expected_output_sin_no_blowup,
    #                         my_new_S_operator_output_sin)
    #     # USE the overloaded functions to compute more complex integrals
    #     # - TWO radial parts
    #     my_overload_rad_S_operator = my_new_obj._overloaded_symmetry_determiner(
    #         ['hr', 'hr', 'hr'], [['_y_1', '_y_1', '_y_1'], ['_y_1', '_y_1', '_y_1']],
    #         radial_multiplier=1.0, div_by_sin=[False, False],
    #         ang_factor=[1.0, 1.0], rad_factor=[1.0, 1.0], rad_indexer=np.s_[:]            
    #         )
    #     assert math.isclose(
    #         my_overload_rad_S_operator, my_new_expected_output * 2.0
    #         )
    #     # - TWO ANGULAR PARTS
    #     my_overload_ang_S_operator = my_new_obj._overloaded_symmetry_determiner(
    #         [['hr', 'hr', 'hr'], ['hr', 'hr', 'hr']], ['_y_1', '_y_1', '_y_1'],
    #         radial_multiplier=1.0, div_by_sin=[False, False],
    #         ang_factor=[1.0, 1.0], rad_factor=[1.0, 1.0], rad_indexer=np.s_[:]            
    #         )
    #     assert math.isclose(
    #         my_overload_ang_S_operator, my_new_expected_output * 2.0
    #         )        
    #     # - TWO RADIAL AND TWO ANGULAR PARTS
    #     my_overload_rad_ang_S_operator = my_new_obj._overloaded_symmetry_determiner(
    #         [['hr', 'hr', 'hr'], ['hr', 'hr', 'hr']],
    #         [['_y_1', '_y_1', '_y_1'], ['_y_1', '_y_1', '_y_1']],
    #         radial_multiplier=1.0, div_by_sin=[False, False],
    #         ang_factor=[1.0, 1.0], rad_factor=[1.0, 1.0], rad_indexer=np.s_[:]            
    #         )
    #     assert math.isclose(
    #         my_overload_rad_ang_S_operator, my_new_expected_output * 4.0
    #         )
    #     # rad factor
    #     my_overload_rad_ang_S_operator_rad_fac = my_new_obj._overloaded_symmetry_determiner(
    #         [['hr', 'hr', 'hr'], ['hr', 'hr', 'hr']],
    #         [['_y_1', '_y_1', '_y_1'], ['_y_1', '_y_1', '_y_1']],
    #         radial_multiplier=1.0, div_by_sin=[False, False],
    #         ang_factor=[1.0, 1.0], rad_factor=[0.5, 3.0], rad_indexer=np.s_[:]            
    #         ) 
    #     assert math.isclose(
    #         my_overload_rad_ang_S_operator_rad_fac, my_new_expected_output * 7.0
    #         )       
    #     # ang factor
    #     my_overload_rad_ang_S_operator_ang_fac = my_new_obj._overloaded_symmetry_determiner(
    #         [['hr', 'hr', 'hr'], ['hr', 'hr', 'hr']],
    #         [['_y_1', '_y_1', '_y_1'], ['_y_1', '_y_1', '_y_1']],
    #         radial_multiplier=1.0, div_by_sin=[False, False],
    #         ang_factor=[0.5, 3.0], rad_factor=[1.0, 1.0], rad_indexer=np.s_[:]            
    #         ) 
    #     assert math.isclose(
    #         my_overload_rad_ang_S_operator_ang_fac, my_new_expected_output * 7.0
    #         )   
    #     # combination
    #     my_overload_rad_ang_S_operator_combo = my_new_obj._overloaded_symmetry_determiner(
    #         [['hr', 'hr', 'hr'], ['hr', 'hr', 'hr']],
    #         [['_y_1', '_y_1', '_y_1'], ['_y_1', '_y_1', '_y_1']],
    #         radial_multiplier=1.0, div_by_sin=[True, False],
    #         ang_factor=[1.0, 1.0], rad_factor=[1.0, 1.0], rad_indexer=np.s_[:]            
    #         )  
    #     assert math.isclose(
    #         my_overload_rad_ang_S_operator_combo,
    #         (my_new_expected_output * 2.0) + (my_expected_output_sin_no_blowup * 2.0)
    #         )                     
    #     # combination 2
    #     my_overload_rad_ang_S_operator_combo2 = my_new_obj._overloaded_symmetry_determiner(
    #         [['hr', 'hr', 'hr'], ['hr', 'hr', 'hr']],
    #         [['_y_1', '_y_1', '_y_1'], ['_y_1', '_y_1', '_y_1']],
    #         radial_multiplier=1.0, div_by_sin=[False, False],
    #         ang_factor=[0.25, 1.0], rad_factor=[1.0, 1.0], rad_indexer=np.s_[:]            
    #         )
    #     assert math.isclose(
    #         my_overload_rad_ang_S_operator_combo2,
    #         (my_new_expected_output * 2.5)
    #         )  
    #     # combination 3
    #     my_overload_rad_ang_S_operator_combo3 = my_new_obj._overloaded_symmetry_determiner(
    #         [['hr', 'hr', 'hr'], ['hr', 'hr', 'hr']],
    #         [['_y_1', '_y_1', '_y_1'], ['_y_1', '_y_1', '_y_1']],
    #         radial_multiplier=1.0, div_by_sin=[False, False],
    #         ang_factor=[0.25, 1.0], rad_factor=[1.5, 1.0], rad_indexer=np.s_[:]            
    #         )
    #     # == (1.0 + 1.5 + 0.375 + 0.25) * expected
    #     assert math.isclose(
    #         my_overload_rad_ang_S_operator_combo3,
    #         (my_new_expected_output * 3.125)
    #         )  
    #     # combination 4
    #     my_overload_rad_ang_S_operator_combo4 = my_new_obj._overloaded_symmetry_determiner(
    #         [['hr', 'hr', 'hr'], ['hr', 'hr', 'hr']],
    #         [['_y_1', '_y_1', '_y_1'], ['_y_1', '_y_1', '_y_1']],
    #         radial_multiplier=2.0, div_by_sin=[False, False],
    #         ang_factor=[0.25, 1.0], rad_factor=[1.5, 1.0], rad_indexer=np.s_[:]            
    #         )
    #     # == (1.0 + 1.5 + 0.375 + 0.25) * expected * 2.0 (radial multiplier)
    #     assert math.isclose(
    #         my_overload_rad_ang_S_operator_combo4,
    #         (my_new_expected_output * 3.125 * 2.0)
    #         ) 
    #     # combination 5
    #     my_overload_rad_ang_S_operator_combo5 = my_new_obj._overloaded_symmetry_determiner(
    #         [['hr', 'hr', 'hr'], ['hr', 'hr', 'hr']],
    #         [['_y_1', '_y_1', '_y_1'], ['_y_1', '_y_1', '_y_1']],
    #         radial_multiplier=my_new_obj._x.copy(), div_by_sin=[False, False],
    #         ang_factor=[0.25, 1.0], rad_factor=[1.5, 1.0], rad_indexer=np.s_[:]            
    #         )
    #     # my expected radial CHANGES because each value is effectively multiplied
    #     # with the corresponding x value(!):
    #     my_newest_expected_radial = 1.674530029296875
    #     my_newest_expected_output = (my_newest_expected_radial * 6.0) * my_expected_angular
    #     # == (1.0 + 1.5 + 0.375 + 0.25) * expected
    #     assert math.isclose(
    #         my_overload_rad_ang_S_operator_combo5,
    #         (my_newest_expected_output * 3.125)
    #         ) 

    # def test_ccr_zero_azimuthal(self):
    #     """Tests whether the coupling coefficient is zero when
    #     the azimuthal selection rule is not fulfilled.
    #     """
    #     # initialize the object
    #     my_obj = QCCR(
    #         **self.input_value_object,
    #         freq_handler=self.my_fh
    #         )
    #     # compute the coupling coefficient
    #     my_obj.adiabatic_coupling_rot(indexer=np.s_[:])
    #     # assert that the coupling coefficient is zero
    #     assert math.isclose(
    #         my_obj.coupling_coefficient,
    #         0.0
    #         )    
    #     # initialize a second object
    #     my_second_obj = QCCR(
    #         **self.input_value_object,
    #         freq_handler=self.my_fh
    #         )
    #     # compute the coupling coefficient
    #     my_second_obj.adiabatic_coupling_rot(indexer=np.s_[1:])
    #     # assert that the coupling coefficient is zero
    #     assert math.isclose(
    #         my_second_obj.coupling_coefficient,
    #         0.0
    #         ) 
    #     # change the azimuthal wavenumbers so that
    #     # we obtain a non-zero coupling coefficient for conj input
    #     my_input_value_object_to_be_changed = self.input_value_object.copy()
    #     my_input_value_object_to_be_changed['m_list'] = [-2, 1, 1]  # Azimuthal selection rule NOT fulfilled  
    #     # initialize the object
    #     my_third_obj = QCCR(
    #         **my_input_value_object_to_be_changed,
    #         freq_handler=self.my_fh
    #         )        
    #     # compute the coupling coefficient        
    #     my_third_obj.adiabatic_coupling_rot()
    #     # assert that the coupling coefficient is zero, because we are using conjugated modes
    #     assert math.isclose(
    #         my_third_obj.coupling_coefficient,
    #         0.0
    #         )   

    # def test_ccr_zero_meridional(self):
    #     """Tests whether the coupling coefficient is zero when
    #     the meridional selection rule is not fulfilled.
    #     """
    #      # change the meridional orders by changing the 'spherical degrees'
    #     my_input_value_object_to_be_changed = self.input_value_object.copy()
    #     my_input_value_object_to_be_changed['m_list'] = [2, 1, 1]  # Azimuthal selection rule fulfilled
    #     my_input_value_object_to_be_changed['m_list'] = [2, 2, 3]  # Meridional selection rule NOT fulfilled
    #     # initialize the object
    #     my_obj = QCCR(
    #         **my_input_value_object_to_be_changed,
    #         freq_handler=self.my_fh
    #         )        
    #     # compute the coupling coefficient
    #     my_obj.adiabatic_coupling_rot(indexer=np.s_[:])
    #     # assert that the coupling coefficient is zero
    #     assert math.isclose(
    #         my_obj.coupling_coefficient,
    #         0.0
    #         )  
    #     # make a second object
    #     my_second_obj = QCCR(
    #         **my_input_value_object_to_be_changed,
    #         freq_handler=self.my_fh
    #         )
    #     # ensure that the coupling coefficient is also zero
    #     # when the indexer is used
    #     my_second_obj.adiabatic_coupling_rot(indexer=np.s_[1:])
    #     # assert that the coupling coefficient is zero
    #     assert math.isclose(
    #         my_second_obj.coupling_coefficient,
    #         0.0
    #         )

    # def test_values_terms_qccr(self):
    #     """Tests the values of the individual terms
    #     of the coupling coefficient.
    #     """
    #     # change the azimuthal wavenumbers so that
    #     # we obtain a non-zero coupling coefficient
    #     my_input_value_object_to_be_changed = self.input_value_object.copy()
    #     my_input_value_object_to_be_changed['m_list'] = [2, 1, 1]  # Azimuthal selection rule fulfilled        
    #     # initialize the object
    #     my_obj = QCCR(
    #         **my_input_value_object_to_be_changed,
    #         freq_handler=self.my_fh, store_debug=True
    #         )
    #     # assert that the individual terms are equal to what we expect
    #     assert math.isclose(
    #         my_obj._kappa_1_abc(indexer=np.s_[:]), 0.0
    #         )
    #     assert math.isclose(
    #         my_obj._kappa_2_abc(indexer=np.s_[:]), 8.989663047961028e+30
    #         )
    #     assert math.isclose(
    #         my_obj._kappa_3_abc(indexer=np.s_[:]), -2.702424016541382e+36
    #         )
    #     assert math.isclose(
    #         my_obj._kappa_4_abc(indexer=np.s_[:]), 1.342715526481236e+60
    #         )
    
    # def test_value_qccr(self):
    #     """Tests the value of the coupling coefficient.
    #     """
    #     # change the azimuthal wavenumbers so that
    #     # we obtain a non-zero coupling coefficient
    #     my_input_value_object_to_be_changed = self.input_value_object.copy()
    #     my_input_value_object_to_be_changed['m_list'] = [2, 1, 1]  # Azimuthal selection rule fulfilled        
    #     # initialize the object
    #     my_obj = QCCR(
    #         **my_input_value_object_to_be_changed,
    #         freq_handler=self.my_fh
    #         )
    #     # compute the coupling coefficient        
    #     my_obj.adiabatic_coupling_rot()
    #     # assert that the coupling coefficient is equal to the sum
    #     # of the individual terms
    #     expected_value_cc = 1.342715526481236e+60 - 2.702424016541382e+36\
    #         + 8.989663047961028e+30
    #     mode_energy = 1.5017175e+49
    #     assert math.isclose(
    #         my_obj.coupling_coefficient, expected_value_cc
    #         )
    #     # assert that the normalized coupling coefficient value is ok
    #     assert math.isclose(
    #         my_obj.normed_coupling_coefficient,
    #         expected_value_cc / mode_energy
    #         )
    #     # assert that the normalized coupling coefficient defined as in 
    #     # Lee (2012) is ok
    #     assert math.isclose(
    #         my_obj.eta_lee_coupling_coefficient,
    #         2.0 * expected_value_cc / mode_energy
    #         )

    # def test_conj_ccr_zero_azimuthal(self):
    #     """Tests whether the coupling coefficient is zero when
    #     the azimuthal selection rule is not fulfilled if a eigenfunction
    #     is complex conjugated in the coupling coefficient.
    #     This differs from the standard coupling coefficients we compute.
    #     """
    #     # initialize the object
    #     my_obj = QCCR(
    #         **self.input_value_object,
    #         freq_handler=self.my_fh,
    #         conj_cc=[False, False, False]
    #         )
    #     # compute the coupling coefficient
    #     my_obj.adiabatic_coupling_rot(indexer=np.s_[:])
    #     # assert that the coupling coefficient is zero
    #     assert math.isclose(
    #         my_obj.coupling_coefficient,
    #         0.0
    #         )    
    #     # initialize a second object
    #     my_second_obj = QCCR(
    #         **self.input_value_object,
    #         freq_handler=self.my_fh,
    #         conj_cc=[False, False, False]
    #         )
    #     # compute the coupling coefficient
    #     my_second_obj.adiabatic_coupling_rot(indexer=np.s_[1:])
    #     # assert that the coupling coefficient is zero
    #     assert math.isclose(
    #         my_second_obj.coupling_coefficient,
    #         0.0
    #         )  
    #     # change the azimuthal wavenumbers so that
    #     # we obtain a non-zero coupling coefficient for non-conj input
    #     my_input_value_object_to_be_changed = self.input_value_object.copy()
    #     my_input_value_object_to_be_changed['m_list'] = [2, 1, 1]  # Azimuthal selection rule NOT fulfilled  
    #     # initialize the object
    #     my_third_obj = QCCR(
    #         **my_input_value_object_to_be_changed,
    #         freq_handler=self.my_fh,
    #         conj_cc=[False, False, False]
    #         )        
    #     # compute the coupling coefficient        
    #     my_third_obj.adiabatic_coupling_rot()
    #     # assert that the coupling coefficient is zero, because we are using conjugated modes
    #     assert math.isclose(
    #         my_third_obj.coupling_coefficient,
    #         0.0
    #         )

    # def test_conj_values_terms_qccr(self):
    #     """Tests the values of the individual terms
    #     of the coupling coefficient if a eigenfunction
    #     is complex conjugated in the coupling coefficient.
    #     """
    #     # change the azimuthal wavenumbers so that
    #     # we obtain a non-zero coupling coefficient
    #     my_input_value_object_to_be_changed = self.input_value_object.copy()
    #     my_input_value_object_to_be_changed['m_list'] = [-2, 1, 1]  # Azimuthal selection rule fulfilled        
    #     # initialize the object
    #     my_obj = QCCR(
    #         **my_input_value_object_to_be_changed,
    #         freq_handler=self.my_fh,
    #         conj_cc=[False, False, False]
    #         )
    #     # assert that the individual terms are equal to what we expect
    #     assert math.isclose(
    #         my_obj._kappa_1_abc(indexer=np.s_[:]), 0.0
    #         )
    #     assert math.isclose(
    #         my_obj._kappa_2_abc(indexer=np.s_[:]), 2.0454575826487366e+31
    #         )
    #     assert math.isclose(
    #         my_obj._kappa_3_abc(indexer=np.s_[:]), -1.7792978221015536e+36
    #         )
    #     assert math.isclose(
    #         my_obj._kappa_4_abc(indexer=np.s_[:]), 3.0551397091620886e+60
    #         )
        
    # def test_conj_value_qccr(self):
    #     """Tests the value of the coupling coefficient if a eigenfunction
    #     is complex conjugated in the coupling coefficient.
    #     """
    #     # change the azimuthal wavenumbers so that
    #     # we obtain a non-zero coupling coefficient
    #     my_input_value_object_to_be_changed = self.input_value_object.copy()
    #     my_input_value_object_to_be_changed['m_list'] = [-2, 1, 1]  # Azimuthal selection rule fulfilled        
    #     # initialize the object
    #     my_obj = QCCR(
    #         **my_input_value_object_to_be_changed,
    #         freq_handler=self.my_fh,
    #         conj_cc=[False, False, False]
    #         )
    #     # compute the coupling coefficient        
    #     my_obj.adiabatic_coupling_rot()
    #     # assert that the coupling coefficient is equal to the sum
    #     # of the individual terms
    #     expected_value_cc = 3.0551397091620886e+60 - 1.7792978221015536e+36\
    #         + 2.0454575826487366e+31
    #     mode_energy = 1.5017175e+49
    #     assert math.isclose(
    #         my_obj.coupling_coefficient, expected_value_cc
    #         )
    #     # assert that the normalized coupling coefficient value is ok
    #     assert math.isclose(
    #         my_obj.normed_coupling_coefficient,
    #         expected_value_cc / mode_energy
    #         )
    #     # assert that the normalized coupling coefficient defined as in 
    #     # Lee (2012) is ok
    #     assert math.isclose(
    #         my_obj.eta_lee_coupling_coefficient,
    #         2.0 * expected_value_cc / mode_energy
    #         )

    # def test_retro_values_terms_qccr(self):
    #     """Tests the values of the individual terms
    #     of the coupling coefficient when using only retrograde modes
    #     in the coupling coefficient.
    #     """
    #     # change the azimuthal wavenumbers so that
    #     # we obtain a non-zero coupling coefficient
    #     my_input_value_object_to_be_changed = self.input_value_object.copy()
    #     my_input_value_object_to_be_changed['m_list'] = [-2, -1, -1]  # Azimuthal selection rule fulfilled        
    #     # initialize the object
    #     my_obj = QCCR(
    #         **my_input_value_object_to_be_changed,
    #         freq_handler=self.my_fh,
    #         )
    #     # assert that the individual terms are equal to what we expect
    #     assert math.isclose(
    #         my_obj._kappa_1_abc(indexer=np.s_[:]), 0.0
    #         )
    #     assert math.isclose(
    #         my_obj._kappa_2_abc(indexer=np.s_[:]), 2.8624654067416657e+31
    #         )
    #     assert math.isclose(
    #         my_obj._kappa_3_abc(indexer=np.s_[:]), 2.493523151639402e+36
    #         )
    #     assert math.isclose(
    #         my_obj._kappa_4_abc(indexer=np.s_[:]), 4.2754402752829464e+60
    #         )

    # def test_retro_value_qccr(self):
    #     """Tests the value of the coupling coefficient using only
    #     retrograde modes in the coupling coefficient.
    #     """
    #     # change the azimuthal wavenumbers so that
    #     # we obtain a non-zero coupling coefficient
    #     my_input_value_object_to_be_changed = self.input_value_object.copy()
    #     my_input_value_object_to_be_changed['m_list'] = [-2, -1, -1]  # Azimuthal selection rule fulfilled        
    #     # initialize the object
    #     my_obj = QCCR(
    #         **my_input_value_object_to_be_changed,
    #         freq_handler=self.my_fh
    #         )
    #     # compute the coupling coefficient        
    #     my_obj.adiabatic_coupling_rot()
    #     # assert that the coupling coefficient is equal to the sum
    #     # of the individual terms
    #     expected_value_cc = 4.2754402752829464e+60 + 2.493523151639402e+36\
    #         + 2.8624654067416657e+31
    #     mode_energy = 1.5017175e+49
    #     assert math.isclose(
    #         my_obj.coupling_coefficient, expected_value_cc
    #         )
    #     # assert that the normalized coupling coefficient value is ok
    #     assert math.isclose(
    #         my_obj.normed_coupling_coefficient,
    #         expected_value_cc / mode_energy
    #         )
    #     # assert that the normalized coupling coefficient defined as in 
    #     # Lee (2012) is ok
    #     assert math.isclose(
    #         my_obj.eta_lee_coupling_coefficient,
    #         2.0 * expected_value_cc / mode_energy
    #         )
