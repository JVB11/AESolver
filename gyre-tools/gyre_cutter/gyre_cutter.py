"""Python enumeration module that contains enumeration objects that store data used to transform the data arrays for mode coupling into the correct shapes.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import logging
import sys
import json
import numpy as np
from enum import Enum


# set up logger
logger = logging.getLogger(__name__)


# enumeration class used to store attributes for which center points need to be cut
class EnumerationCutter(Enum):
    """Enumeration class used to store attributes for which center points need to be dropped, in order to avoid having NaN values pop up when computing derivatives near the center of the model."""

    # store enumeration members in the form of json-string-converted lists
    STRUCTURE_COEFFICIENT_CUTS = json.dumps(['_c_1', '_V_2', '_As', '_U'])
    POLYTROPIC_CUTS = json.dumps(
        [
            '_xi_h_mode_1',
            '_xi_h_mode_2',
            '_xi_h_mode_3',
            '_xi_r_mode_1',
            '_xi_r_mode_2',
            '_xi_r_mode_3',
            '_y_1_mode_1',
            '_y_1_mode_2',
            '_y_1_mode_3',
            '_y_2_mode_1',
            '_y_2_mode_2',
            '_y_2_mode_3',
        ]
    )
    MESA_GYRE_CUTS = json.dumps(
        [
            '_deul_phi_mode_1',
            '_deul_phi_mode_2',
            '_deul_phi_mode_3',
            '_eul_phi_mode_1',
            '_eul_phi_mode_2',
            '_eul_phi_mode_3',
            '_xi_h_mode_1',
            '_xi_h_mode_2',
            '_xi_h_mode_3',
            '_xi_r_mode_1',
            '_xi_r_mode_2',
            '_xi_r_mode_3',
            '_y_1_mode_1',
            '_y_1_mode_2',
            '_gamma_1_adDer_profile',
            '_y_1_mode_3',
            '_y_2_mode_1',
            '_y_2_mode_2',
            '_y_2_mode_3',
        ]
    )
    PRE_PROFILE_LOAD_CUTS = json.dumps(
        ['_x', '_M_r', '_rho', '_P', '_Gamma_1', '_brunt_N2_profile']
    )
    PRE_PROFILE_LOAD_CUTS_POLY = json.dumps(
        [
            '_x',
            '_mr_profile',
            '_rho_profile',
            '_r_profile',
            '_p_profile',
            '_Gamma_1',
            '_brunt_n2_profile',
        ]
    )
    PROFILE_LOAD_CUTS_ANALYTIC_POLY = json.dumps(
        [
            '_x',
            '_mr_profile',
            '_rho_profile',
            '_r_profile',
            '_p_profile',
            '_Gamma_1',
            '_brunt_n2_profile',
            '_drho',
            '_drho2',
            '_drho3',
            '_dP',
            '_dP2',
            '_dP3',
            '_dgravpot',
            '_dgravpot2',
            '_dgravpot3',
        ]
    )

    # generic method used for cutting
    @classmethod
    def cutter(cls, my_obj, cut_list, cut_slice=np.s_[1:], polytrope=False):
        """Internal generic cutting method.

        Parameters
        ----------
        my_obj : CouplingCoefficients
            The CouplingCoefficients object, whose attributes need their center points removed.
        cut_list : list[str] | str
            List of attributes that need their center points removed.
        cut_slice : np.s_ | int, optional
            The slicing action for the cutter; by default cut the center point --> np.s_[1:].
        polytrope : bool, optional
            If True, get attributes for polytrope computations. If False, get attributes for MESA+GYRE computations; by default False.
        """
        # check if 'cut_list' is a string, and if so, obtain the specific list it refers to
        try:
            # convert to possible polytrope equivalent if necessary
            my_cut_string = f"{cut_list.upper()}{'_POLY' if polytrope else ''}"
            # cut_list is a key string that refers to a json string from which the stored list can be recovered
            my_cut_list = json.loads(cls.__getitem__(my_cut_string).value)
        except AttributeError:
            # cut list is expected to be a list
            # - convert to polytrope equivalent if needed
            if polytrope:
                my_cut_list = [f'{c}_POLY' for c in cut_list]
            else:
                my_cut_list = cut_list
        except KeyError:
            # the key string does not exist as an enumeration member
            logger.exception(
                f'Enumeration member {cut_list} does not exist. Now exiting.'
            )
            sys.exit()
        # cut points of attributes whose keys are mentioned in the cut_list; DEFAULT behavior = cut center points (i.e. cut_slice = np.s_[1:])
        for _c in my_cut_list:
            setattr(my_obj, _c, getattr(my_obj, _c)[cut_slice])
