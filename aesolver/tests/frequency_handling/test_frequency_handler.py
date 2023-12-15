'''Pytest module for the frequency handler coefficient support class of the aesolver.coupling_coefficients module.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
'''
# import statements
import pytest
import typing
import numpy as np
# import support modules
from mock_gyre_dicts import \
    MockGYREAdiabaticInputObject as MGAIo
from mock_gyre_dicts import \
    MockGYRENonAdiabaticInputObject as MGnAIo
from pytest_util_functions import \
    value_comparison, compare_elements
# get expected value modules
from expected_freq_handler_values import \
    ExpectedValuesFreqHandlerAdiabatic as EVFHa
from expected_freq_handler_values import \
    ExpectedValuesFreqHandlerNonAdiabatic as EVFHna
# import module to be tested
from aesolver.frequency_handling import FH


# fixture for the test class
@pytest.fixture(scope='class')
def setup_test_class(request) -> None:
    # define the request units for our calculations
    request.cls.requested_units = 'HZ'
    # define the adiabatic GYRE input dictionary list
    request.cls.gyre_ad_input_dicts = \
        MGAIo.get_gyre_dicts()
    # define the non-adiabatic GYRE input dictionary list
    request.cls.gyre_nad_input_dicts = \
        MGnAIo.get_gyre_dicts()


# define the test class
@pytest.mark.usefixtures('setup_test_class')
class TestFreqHandlerObject:
    """Tests the implementation of the frequency handler coefficient support class of the aesolver.coupling_coefficients module."""
    
    # attribute type declarations
    requested_units: np.bytes_
    gyre_ad_input_dicts: list[dict[str, typing.Any]]
    gyre_nad_input_dicts: list[dict[str, typing.Any]]

    def test_initialization_freq_handler(self):
        """Tests the initialization of the frequency
        handler object.
        """
        # create a new frequency handler object
        my_obj = FH(*self.gyre_ad_input_dicts,
                    my_freq_units=self.requested_units)
        # assert that the object is made
        assert my_obj
        
    def test_adiabatic_attributes_freq_handler(self):
        """Tests whether the adiabatic attributes of the frequency handler object are as expected."""
        # create a new frequency handler object
        my_obj = FH(*self.gyre_ad_input_dicts,
                    my_freq_units=self.requested_units)
        # get the expected value dict
        evd = EVFHa.get_dict()
        # assert that the adiabatic attributes
        # are loaded correctly
        assert my_obj.requested_units == evd['_requested_units']
        assert my_obj.nr_of_modes == evd['_n_modes']
        assert my_obj.angular == evd['_angular']
        assert value_comparison(
            my_obj.surf_rot_freq, evd['_rot_freq']
            )
        assert value_comparison(
            my_obj.surf_rot_angular_freq, evd['_rot_freq_ang']
            )
        assert value_comparison(
            my_obj.surf_rot_angular_freq_rps, evd['_rot_freq_ang_rps']
            )
        assert value_comparison(
            my_obj.angular_conv_factor_from_rps, evd['_ang_conv_rps']
            ) 
        assert compare_elements(
            my_obj._cffs, evd['_cffs']
            )
        assert value_comparison(
            my_obj._ocfr, evd['_ocfr']
            )
        assert value_comparison(
            my_obj._fcfr, evd['_fcfr']
            )
        assert value_comparison(
            my_obj._ocfc, evd['_ocfc']
            )
        assert value_comparison(
            my_obj._fcfc, evd['_fcfc']
            )  
        assert my_obj._om_req == evd['_om_req']       
        assert np.allclose(
            my_obj.inert_mode_freqs,
            evd['_inert_mode_freqs']
            )      
        assert np.allclose(
            my_obj.inert_mode_omegas,
            evd['_inert_mode_omegas']
            )
        assert np.allclose(
            my_obj.corot_mode_freqs,
            evd['_corot_mode_freqs']
            )
        assert np.allclose(
            my_obj.corot_mode_omegas,
            evd['_corot_mode_omegas']
            )
        assert np.allclose(
            my_obj.spin_factors,
            evd['_spin_factors']
            )
        assert np.allclose(
            my_obj.dimless_inert_freqs,
            evd['_dimless_inert_freqs']
            )      
        assert np.allclose(
            my_obj.dimless_inert_omegas,
            evd['_dimless_inert_omegas']
            )
        assert np.allclose(
            my_obj.dimless_corot_freqs,
            evd['_dimless_corot_freqs']
            )
        assert np.allclose(
            my_obj.dimless_corot_omegas,
            evd['_dimless_corot_omegas']
            )
        assert value_comparison(
            my_obj.detuning,
            evd['_detuning']
            )

    def test_nad_init_freq_handler(self):
        """Tests the initialization of the frequency handler object."""
        # create a new frequency handler object
        my_obj = FH(*self.gyre_ad_input_dicts,
                    my_freq_units=self.requested_units)
        # initialize the non-adiabatic information
        my_obj.set_mode_nonadiabatic_info(
            *self.gyre_nad_input_dicts
            )
        # assert the initialization went fine
        assert my_obj

    def test_nad_attributes_freq_handler(self):
        """Tests whether the non-adiabatic attributes of the frequency handler object are as expected."""
        # create a new frequency handler object
        my_obj = FH(*self.gyre_ad_input_dicts,
                    my_freq_units=self.requested_units)
        # initialize the non-adiabatic information
        my_obj.set_mode_nonadiabatic_info(
            *self.gyre_nad_input_dicts
            )
        # get the expected value dictionary
        evd = EVFHna.get_dict()
        # assert that the nonadiabatic attributes
        # are loaded correctly
        assert np.allclose(
            my_obj.driving_rates,
            evd['_driving_rates']
            )
        assert np.allclose(
            my_obj.inert_mode_freqs_nad,
            evd['_inert_mode_freqs_nad']
            )
        assert np.allclose(
            my_obj.inert_mode_omegas_nad,
            evd['_inert_mode_omegas_nad']
            )
        assert np.allclose(
            my_obj.corot_mode_freqs_nad,
            evd['_corot_mode_freqs_nad']
            )
        assert np.allclose(
            my_obj.corot_mode_omegas_nad,
            evd['_corot_mode_omegas_nad']
            )        
        assert np.allclose(
            my_obj.quality_factors,
            evd['_quality_factors']
            )
        assert np.allclose(
            my_obj.quality_factors_nad,
            evd['_quality_factors_nad']
            )
        assert np.allclose(
            my_obj.quality_factor_products,
            evd['_quality_factor_products']
            )
        assert np.allclose(
            my_obj.quality_factor_products_nad,
            evd['_quality_factor_products_nad']
            )
        assert value_comparison(
            my_obj.gamma_sum,
            evd['_gamma_sum']
            )
        assert np.allclose(
            my_obj.dimless_inert_freqs_nad,
            evd['_dimless_inert_freqs_nad']
            )
        assert np.allclose(
            my_obj.dimless_inert_omegas_nad,
            evd['_dimless_inert_omegas_nad']
            )
        assert np.allclose(
            my_obj.dimless_corot_freqs_nad,
            evd['_dimless_corot_freqs_nad']
            )
        assert np.allclose(
            my_obj.dimless_corot_omegas_nad,
            evd['_dimless_corot_omegas_nad']
            ) 
        assert value_comparison(
            my_obj.detuning_nad,
            evd['_detuning_nad']
            )
        assert value_comparison(
            my_obj.q_factor,
            evd['_q_fac']
            )
        assert value_comparison(
            my_obj.q_factor_nad,
            evd['_q_fac_nad']
            )
        assert value_comparison(
            my_obj.q1_factor,
            evd['_q1_fac']
            )
        assert value_comparison(
            my_obj.q1_factor_nad,
            evd['_q1_fac_nad']
            )

    def test_get_inert_corot_f(self):
        """Tests whether the 'get_inert_corot_f' methods of the frequency handler object produces  expected output."""
        # create a new frequency handler object
        my_obj = FH(*self.gyre_ad_input_dicts,
                    my_freq_units=self.requested_units)
        # initialize the non-adiabatic information
        my_obj.set_mode_nonadiabatic_info(
            *self.gyre_nad_input_dicts
            )
        # get the expected value dictionary
        evd = EVFHa.get_dict()
        evd_nad = EVFHna.get_dict()
        # get the adiabatic output from the method
        out_f_1 = my_obj.get_inert_corot_f(my_f_nr=1)
        out_f_2 = my_obj.get_inert_corot_f(my_f_nr=2)
        out_f_3 = my_obj.get_inert_corot_f(my_f_nr=3)
        # assert this output is correct
        assert out_f_1[0] == evd['_corot_mode_freqs'][0]
        assert out_f_1[1] == evd['_inert_mode_freqs'][0]
        assert out_f_2[0] == evd['_corot_mode_freqs'][1]
        assert out_f_2[1] == evd['_inert_mode_freqs'][1]
        assert out_f_3[0] == evd['_corot_mode_freqs'][2]
        assert out_f_3[1] == evd['_inert_mode_freqs'][2]
        # get the nonadiabatic output from the method
        out_f_1_nad = my_obj.get_inert_corot_f_nad(my_f_nr=1)
        out_f_2_nad = my_obj.get_inert_corot_f_nad(my_f_nr=2)
        out_f_3_nad = my_obj.get_inert_corot_f_nad(my_f_nr=3)
        # assert this output is correct        
        assert out_f_1_nad[0] == \
            evd_nad['_corot_mode_freqs_nad'][0]
        assert out_f_1_nad[1] == \
            evd_nad['_inert_mode_freqs_nad'][0]
        assert out_f_2_nad[0] == \
            evd_nad['_corot_mode_freqs_nad'][1]
        assert out_f_2_nad[1] == \
            evd_nad['_inert_mode_freqs_nad'][1]
        assert out_f_3_nad[0] == \
            evd_nad['_corot_mode_freqs_nad'][2]
        assert out_f_3_nad[1] == \
            evd_nad['_inert_mode_freqs_nad'][2]

    def test_get_inert_corot_o(self):
        """Tests whether the 'get_inert_corot_o' methods of the frequency handler object produces  expected output."""
        # create a new frequency handler object
        my_obj = FH(*self.gyre_ad_input_dicts,
                    my_freq_units=self.requested_units)
        # initialize the non-adiabatic information
        my_obj.set_mode_nonadiabatic_info(
            *self.gyre_nad_input_dicts
            )
        # get the expected value dictionary
        evd = EVFHa.get_dict()
        evd_nad = EVFHna.get_dict()
        # get the adiabatic output from the method
        out_o_1 = my_obj.get_inert_corot_o(my_f_nr=1)
        out_o_2 = my_obj.get_inert_corot_o(my_f_nr=2)
        out_o_3 = my_obj.get_inert_corot_o(my_f_nr=3)
        # assert this output is correct
        assert out_o_1[0] == evd['_corot_mode_omegas'][0]
        assert out_o_1[1] == evd['_inert_mode_omegas'][0]
        assert out_o_2[0] == evd['_corot_mode_omegas'][1]
        assert out_o_2[1] == evd['_inert_mode_omegas'][1]
        assert out_o_3[0] == evd['_corot_mode_omegas'][2]
        assert out_o_3[1] == evd['_inert_mode_omegas'][2]
        # get the nonadiabatic output from the method
        out_o_1_nad = my_obj.get_inert_corot_o_nad(my_f_nr=1)
        out_o_2_nad = my_obj.get_inert_corot_o_nad(my_f_nr=2)
        out_o_3_nad = my_obj.get_inert_corot_o_nad(my_f_nr=3)
        # assert this output is correct        
        assert out_o_1_nad[0] == \
            evd_nad['_corot_mode_omegas_nad'][0]
        assert out_o_1_nad[1] == \
            evd_nad['_inert_mode_omegas_nad'][0]
        assert out_o_2_nad[0] == \
            evd_nad['_corot_mode_omegas_nad'][1]
        assert out_o_2_nad[1] == \
            evd_nad['_inert_mode_omegas_nad'][1]
        assert out_o_3_nad[0] == \
            evd_nad['_corot_mode_omegas_nad'][2]
        assert out_o_3_nad[1] == \
            evd_nad['_inert_mode_omegas_nad'][2]

    def test_get_detunings(self):
        """Tests whether the 'get_detunings' methods of the frequency handler object produces  expected output."""
        # create a new frequency handler object
        my_obj = FH(*self.gyre_ad_input_dicts,
                    my_freq_units=self.requested_units)
        # initialize the non-adiabatic information
        my_obj.set_mode_nonadiabatic_info(
            *self.gyre_nad_input_dicts
            )
        # get the expected value dictionary
        evd = EVFHa.get_dict()
        evd_nad = EVFHna.get_dict()
        # get the adiabatic detuning
        out_ad = my_obj.get_detunings()
        # assert this is correct output
        assert out_ad[0] == evd['_detuning']
        assert out_ad[1] == evd['_detuning']
        # get the nonadiabatic detuning
        out_nad = my_obj.get_detunings_nad()
        # assert this is correct output
        assert out_nad[0] == evd_nad['_detuning_nad']
        assert out_nad[1] == evd_nad['_detuning_nad']

    def test_get_dimless_freqs(self):
        """Tests whether the 'get_dimless_freqs' methods of the frequency handler object produces  expected output."""
        # create a new frequency handler object
        my_obj = FH(*self.gyre_ad_input_dicts,
                    my_freq_units=self.requested_units)
        # initialize the non-adiabatic information
        my_obj.set_mode_nonadiabatic_info(
            *self.gyre_nad_input_dicts
            )
        # get the expected value dictionary
        evd = EVFHa.get_dict()
        evd_nad = EVFHna.get_dict()
        # get the adiabatic output from the method
        out_f_1 = my_obj.get_dimless_freqs(my_f_nr=1)
        out_f_2 = my_obj.get_dimless_freqs(my_f_nr=2)
        out_f_3 = my_obj.get_dimless_freqs(my_f_nr=3)
        # assert this output is correct
        assert out_f_1[0] == evd['_dimless_corot_freqs'][0]
        assert out_f_1[1] == evd['_dimless_inert_freqs'][0]
        assert out_f_2[0] == evd['_dimless_corot_freqs'][1]
        assert out_f_2[1] == evd['_dimless_inert_freqs'][1]
        assert out_f_3[0] == evd['_dimless_corot_freqs'][2]
        assert out_f_3[1] == evd['_dimless_inert_freqs'][2]
        # get the nonadiabatic output from the method
        out_f_1_nad = my_obj.get_dimless_freqs_nad(my_f_nr=1)
        out_f_2_nad = my_obj.get_dimless_freqs_nad(my_f_nr=2)
        out_f_3_nad = my_obj.get_dimless_freqs_nad(my_f_nr=3)
        # assert this output is correct        
        assert out_f_1_nad[0] == \
            evd_nad['_dimless_corot_freqs_nad'][0]
        assert out_f_1_nad[1] == \
            evd_nad['_dimless_inert_freqs_nad'][0]
        assert out_f_2_nad[0] == \
            evd_nad['_dimless_corot_freqs_nad'][1]
        assert out_f_2_nad[1] == \
            evd_nad['_dimless_inert_freqs_nad'][1]
        assert out_f_3_nad[0] == \
            evd_nad['_dimless_corot_freqs_nad'][2]
        assert out_f_3_nad[1] == \
            evd_nad['_dimless_inert_freqs_nad'][2]

    def test_get_dimless_omegas(self):
        """Tests whether the 'get_dimless_omegas' methods of the frequency handler object produces  expected output."""
        # create a new frequency handler object
        my_obj = FH(*self.gyre_ad_input_dicts,
                    my_freq_units=self.requested_units)
        # initialize the non-adiabatic information
        my_obj.set_mode_nonadiabatic_info(
            *self.gyre_nad_input_dicts
            )
        # get the expected value dictionary
        evd = EVFHa.get_dict()
        evd_nad = EVFHna.get_dict()
        # get the adiabatic output from the method
        out_o_1 = my_obj.get_dimless_omegas(my_f_nr=1)
        out_o_2 = my_obj.get_dimless_omegas(my_f_nr=2)
        out_o_3 = my_obj.get_dimless_omegas(my_f_nr=3)
        # assert this output is correct
        assert out_o_1[0] == evd['_dimless_corot_omegas'][0]
        assert out_o_1[1] == evd['_dimless_inert_omegas'][0]
        assert out_o_2[0] == evd['_dimless_corot_omegas'][1]
        assert out_o_2[1] == evd['_dimless_inert_omegas'][1]
        assert out_o_3[0] == evd['_dimless_corot_omegas'][2]
        assert out_o_3[1] == evd['_dimless_inert_omegas'][2]
        # get the nonadiabatic output from the method
        out_o_1_nad = my_obj.get_dimless_omegas_nad(my_f_nr=1)
        out_o_2_nad = my_obj.get_dimless_omegas_nad(my_f_nr=2)
        out_o_3_nad = my_obj.get_dimless_omegas_nad(my_f_nr=3)
        # assert this output is correct        
        assert out_o_1_nad[0] == \
            evd_nad['_dimless_corot_omegas_nad'][0]
        assert out_o_1_nad[1] == \
            evd_nad['_dimless_inert_omegas_nad'][0]
        assert out_o_2_nad[0] == \
            evd_nad['_dimless_corot_omegas_nad'][1]
        assert out_o_2_nad[1] == \
            evd_nad['_dimless_inert_omegas_nad'][1]
        assert out_o_3_nad[0] == \
            evd_nad['_dimless_corot_omegas_nad'][2]
        assert out_o_3_nad[1] == \
            evd_nad['_dimless_inert_omegas_nad'][2]
