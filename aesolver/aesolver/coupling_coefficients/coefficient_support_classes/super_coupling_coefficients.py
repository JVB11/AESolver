"""Python file containing the superclass of the superclass that defines functionalities for the computation of coupling coefficients.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import copy
import logging
import sys
import typing
import numpy as np

# import intra-package module
from ..enumeration_files import ModeData

# import debug information
from ..enumeration_files.enumerated_debug_properties import (
    DebugMinMaxSuperCouplingCoefficient as DmmScc,
)

# import custom modules and function
from carr_conv import re_im_parts
from gyre_cutter import GYRECutter
from num_deriv import NumericalDifferentiator


# set up logger
logger = logging.getLogger(__name__)


# ignore numpy warnings
np.seterr('ignore')


# ------------------ SUPERCLASS OF MODE COUPLING SUPERCLASSES -------------------
class SuperCouplingCoefficient:
    """Python superclass of superclasses that define the generic methods used to compute coupling coefficients in rotating or non-rotating stars.

    Parameters
    ----------
    nr_modes : int
        The number of modes involved in the computation of the coupling coefficient.
    kwargs_diff_div : dict | None, optional
        The keyword arguments dictionary that sets arguments to be used for computing divergences and their derivatives. If None, no keyword arguments are passed; by default None.
    kwargs_diff_terms : dict | None, optional
        The keyword arguments dictionary that sets arguments to be used for computing derivatives of terms in the coupling coefficient. If None, no keyword arguments are passed; by default None.
    diff_terms_method : str, optional
        Denotes the numerical method used to compute (numerical) derivatives for the terms in the coupling coefficient, if necessary; by default 'gradient'.
    polytropic : bool, optional
        If True, perform the necessary actions to compute the coupling coefficient for polytrope input. If False, assume stellar evolution model input for the computations; by default False.
    cut_center : bool, optional
        If True, cut the center cell value from the stellar evolution or polytropic models. If False, do not cut this value; by default True.
    store_debug : bool, optional
        If True, store debug properties/attributes. If False, do not store these attributes; by default False.
    """

    # attribute type declarations
    _store_debug: bool
    _kwargs_diff_div: dict | None
    _kwargs_diff_terms: dict | None
    _diff_terms_method: str
    _list_common: list[str]
    _use_polytropic: bool
    _analytic_polytrope: bool
    _nr_modes: int
    _cut_center: bool
    _R_star: float
    _G: float
    _use_symbolic_derivative: bool
    _brunt_N2_profile: np.ndarray
    _use_complex: bool
    _G_profile: np.ndarray
    _r_profile: np.ndarray
    _mr_profile: np.ndarray
    _rho_profile: np.ndarray
    _p_profile: np.ndarray
    _brunt_n2_profile: np.ndarray
    _c1_profile: np.ndarray
    _v2_profile: np.ndarray
    _as_profile: np.ndarray
    _u_profile: np.ndarray
    _x: np.ndarray
    _norm_P: np.ndarray | float
    _norm_rho: np.ndarray | float
    _dgravpot: np.ndarray | None
    _drho: np.ndarray | None
    _drho2: np.ndarray | None
    _drho3: np.ndarray | None
    _dgravpot2: np.ndarray | None
    _dgravpot3: np.ndarray | None
    _dP: np.ndarray | None
    _dP2: np.ndarray | None
    _dP3: np.ndarray | None
    _P_normed: np.ndarray
    _dP_normed: np.ndarray | None
    _dP2_normed: np.ndarray | None
    _dP3_normed: np.ndarray | None
    _rho_normed: np.ndarray
    _drho_normed: np.ndarray | None
    _drho2_normed: np.ndarray | None
    _drho3_normed: np.ndarray | None
    _P: np.ndarray
    _rho: np.ndarray
    _Delta_p: float | None
    _Delta_g: float | None
    _As: np.ndarray
    _V_2: np.ndarray
    _U: np.ndarray
    _c_1: np.ndarray
    _Gamma_1: np.ndarray
    _nabla: np.ndarray
    _nabla_ad: np.ndarray
    _dnabla_ad: np.ndarray
    _upsilon_r: np.ndarray | None
    _c_lum: np.ndarray | None
    _c_rad: np.ndarray | None
    _c_thn: np.ndarray | None
    _c_thk: np.ndarray | None
    _c_eps: np.ndarray | None
    _kap_rho: np.ndarray | None
    _kap_T_: np.ndarray | None
    _eps_rho: np.ndarray | None
    _eps_T: np.ndarray | None
    _Omega_rot: np.ndarray
    _M_r: np.ndarray
    _P: np.ndarray
    _rho: np.ndarray
    _T: np.ndarray
    _x: np.ndarray
    # mode-specific attributes
    _n_mode_1: int
    _l_mode_1: int
    _m_mode_1: int
    _G_mode_1: float
    _omega_mode_1: np.ndarray
    _freq_mode_1: np.ndarray
    _freq_units_mode_1: str
    _freq_frame_mode_1: str
    _l_i_mode_1: complex
    _n_p_mode_1: int
    _n_g_mode_1: int
    _n_pg_mode_1: int
    _lambda_mode_1: np.ndarray
    _Omega_rot_dim_mode_1: np.ndarray
    _y_1_mode_1: np.ndarray | None
    _y_2_mode_1: np.ndarray | None
    _y_3_mode_1: np.ndarray | None
    _y_4_mode_1: np.ndarray | None
    _y_5_mode_1: np.ndarray | None
    _y_6_mode_1: np.ndarray | None
    _xi_r_mode_1: np.ndarray | None
    _xi_h_mode_1: np.ndarray | None
    _nabla_mode_1: np.ndarray | None
    _lag_L_mode_1: np.ndarray | None
    _eul_P_mode_1: np.ndarray | None
    _eul_T_mode_1: np.ndarray | None
    _lag_P_mode_1: np.ndarray | None
    _lag_T_mode_1: np.ndarray | None
    _eul_phi_mode_1: np.ndarray | None
    _eul_rho_mode_1: np.ndarray | None
    _lag_rho_mode_1: np.ndarray | None
    _deul_phi_mode_1: np.ndarray | None
    _nabla_ad_mode_1: np.ndarray | None
    _xi_r_dim_mode_1: np.ndarray | None
    _xi_h_dim_mode_1: np.ndarray | None
    _lag_L_dim_mode_1: np.ndarray | None
    _eul_P_dim_mode_1: np.ndarray | None
    _eul_T_dim_mode_1: np.ndarray | None
    _lag_P_dim_mode_1: np.ndarray | None
    _lag_T_dim_mode_1: np.ndarray | None
    _dnabla_ad_mode_1: np.ndarray | None
    _eul_phi_dim_mode_1: np.ndarray | None
    _eul_rho_dim_mode_1: np.ndarray | None
    _lag_rho_dim_mode_1: np.ndarray | None
    _deul_phi_dim_mode_1: np.ndarray | None
    _n_mode_2: int
    _l_mode_2: int
    _m_mode_2: int
    _G_mode_2: float
    _omega_mode_2: np.ndarray
    _freq_mode_2: np.ndarray
    _freq_units_mode_2: str
    _freq_frame_mode_2: str
    _l_i_mode_2: complex
    _n_p_mode_2: int
    _n_g_mode_2: int
    _n_pg_mode_2: int
    _lambda_mode_2: np.ndarray
    _Omega_rot_dim_mode_2: np.ndarray
    _y_1_mode_2: np.ndarray | None
    _y_2_mode_2: np.ndarray | None
    _y_3_mode_2: np.ndarray | None
    _y_4_mode_2: np.ndarray | None
    _y_5_mode_2: np.ndarray | None
    _y_6_mode_2: np.ndarray | None
    _xi_r_mode_2: np.ndarray | None
    _xi_h_mode_2: np.ndarray | None
    _nabla_mode_2: np.ndarray | None
    _lag_L_mode_2: np.ndarray | None
    _eul_P_mode_2: np.ndarray | None
    _eul_T_mode_2: np.ndarray | None
    _lag_P_mode_2: np.ndarray | None
    _lag_T_mode_2: np.ndarray | None
    _eul_phi_mode_2: np.ndarray | None
    _eul_rho_mode_2: np.ndarray | None
    _lag_rho_mode_2: np.ndarray | None
    _deul_phi_mode_2: np.ndarray | None
    _nabla_ad_mode_2: np.ndarray | None
    _xi_r_dim_mode_2: np.ndarray | None
    _xi_h_dim_mode_2: np.ndarray | None
    _lag_L_dim_mode_2: np.ndarray | None
    _eul_P_dim_mode_2: np.ndarray | None
    _eul_T_dim_mode_2: np.ndarray | None
    _lag_P_dim_mode_2: np.ndarray | None
    _lag_T_dim_mode_2: np.ndarray | None
    _dnabla_ad_mode_2: np.ndarray | None
    _eul_phi_dim_mode_2: np.ndarray | None
    _eul_rho_dim_mode_2: np.ndarray | None
    _lag_rho_dim_mode_2: np.ndarray | None
    _deul_phi_dim_mode_2: np.ndarray | None
    _n_mode_3: int
    _l_mode_3: int
    _m_mode_3: int
    _G_mode_3: float
    _omega_mode_3: np.ndarray
    _freq_mode_3: np.ndarray
    _freq_units_mode_3: str
    _freq_frame_mode_3: str
    _l_i_mode_3: complex
    _n_p_mode_3: int
    _n_g_mode_3: int
    _n_pg_mode_3: int
    _lambda_mode_3: np.ndarray
    _Omega_rot_dim_mode_2: np.ndarray
    _y_1_mode_3: np.ndarray | None
    _y_2_mode_3: np.ndarray | None
    _y_3_mode_3: np.ndarray | None
    _y_4_mode_3: np.ndarray | None
    _y_5_mode_3: np.ndarray | None
    _y_6_mode_3: np.ndarray | None
    _xi_r_mode_3: np.ndarray | None
    _xi_h_mode_3: np.ndarray | None
    _nabla_mode_3: np.ndarray | None
    _lag_L_mode_3: np.ndarray | None
    _eul_P_mode_3: np.ndarray | None
    _eul_T_mode_3: np.ndarray | None
    _lag_P_mode_3: np.ndarray | None
    _lag_T_mode_3: np.ndarray | None
    _eul_phi_mode_3: np.ndarray | None
    _eul_rho_mode_3: np.ndarray | None
    _lag_rho_mode_3: np.ndarray | None
    _deul_phi_mode_3: np.ndarray | None
    _nabla_ad_mode_3: np.ndarray | None
    _xi_r_dim_mode_3: np.ndarray | None
    _xi_h_dim_mode_3: np.ndarray | None
    _lag_L_dim_mode_3: np.ndarray | None
    _eul_P_dim_mode_3: np.ndarray | None
    _eul_T_dim_mode_3: np.ndarray | None
    _lag_P_dim_mode_3: np.ndarray | None
    _lag_T_dim_mode_3: np.ndarray | None
    _dnabla_ad_mode_3: np.ndarray | None
    _eul_phi_dim_mode_3: np.ndarray | None
    _eul_rho_dim_mode_3: np.ndarray | None
    _lag_rho_dim_mode_3: np.ndarray | None
    _deul_phi_dim_mode_3: np.ndarray | None
    # attributes for debugging purposes
    _As_max_min: dict[str, float] | None
    _V_2_max_min: dict[str, float] | None
    _U_max_min: dict[str, float] | None
    _c_1_max_min: dict[str, float] | None
    _Gamma_1_max_min: dict[str, float] | None
    _nabla_max_min: dict[str, float] | None
    _nabla_ad_max_min: dict[str, float] | None
    _dnabla_ad_max_min: dict[str, float] | None
    _upsilon_r_max_min: dict[str, float] | None
    _c_lum_max_min: dict[str, float] | None
    _c_rad_max_min: dict[str, float] | None
    _c_thn_max_min: dict[str, float] | None
    _c_thk_max_min: dict[str, float] | None
    _c_eps_max_min: dict[str, float] | None
    _kap_rho_max_min: dict[str, float] | None
    _kap_T__max_min: dict[str, float] | None
    _eps_rho_max_min: dict[str, float] | None
    _eps_T_max_min: dict[str, float] | None
    _Omega_rot_max_min: dict[str, float] | None
    _M_r_max_min: dict[str, float] | None
    _P_max_min: dict[str, float] | None
    _rho_max_min: dict[str, float] | None
    _T_max_min: dict[str, float] | None
    _x_max_min: dict[str, float] | None
    _dgravpot_max_min: dict[str, float] | None
    _dgravpot2_max_min: dict[str, float] | None
    _dgravpot3_max_min: dict[str, float] | None
    _P_max_min: dict[str, float] | None
    _rho_max_min: dict[str, float] | None
    _dP_max_min: dict[str, float] | None
    _dP2_max_min: dict[str, float] | None
    _dP3_max_min: dict[str, float] | None
    _drho_max_min: dict[str, float] | None
    _drho2_max_min: dict[str, float] | None
    _drho3_max_min: dict[str, float] | None
    _P_max_min_normed: dict[str, float] | None
    _rho_max_min_normed: dict[str, float] | None
    _dP_max_min_normed: dict[str, float] | None
    _dP2_max_min_normed: dict[str, float] | None
    _dP3_max_min_normed: dict[str, float] | None
    _drho_max_min_normed: dict[str, float] | None
    _drho2_max_min_normed: dict[str, float] | None
    _drho3_max_min_normed: dict[str, float] | None
    _dgravpot_max_min_poly: dict[str, float] | None
    _dgravpot2_max_min_poly: dict[str, float] | None
    _dgravpot3_max_min_poly: dict[str, float] | None
    _P_max_min_poly: dict[str, float] | None
    _rho_max_min_poly: dict[str, float] | None
    _dP_max_min_poly: dict[str, float] | None
    _dP2_max_min_poly: dict[str, float] | None
    _dP3_max_min_poly: dict[str, float] | None
    _drho_max_min_poly: dict[str, float] | None
    _drho2_max_min_poly: dict[str, float] | None
    _drho3_max_min_poly: dict[str, float] | None
    _dgravpot_max_min_normed_poly: dict[str, float] | None
    _dgravpot2_max_min_normed_poly: dict[str, float] | None
    _dgravpot3_max_min_normed_poly: dict[str, float] | None
    _P_max_min_normed_poly: dict[str, float] | None
    _rho_max_min_normed_poly: dict[str, float] | None
    _dP_max_min_normed_poly: dict[str, float] | None
    _dP2_max_min_normed_poly: dict[str, float] | None
    _dP3_max_min_normed_poly: dict[str, float] | None
    _drho_max_min_normed_poly: dict[str, float] | None
    _drho2_max_min_normed_poly: dict[str, float] | None
    _drho3_max_min_normed_poly: dict[str, float] | None
    _attr_list_max_min: list[str]
    # mode-specific attributes for debugging purposes
    _omega_mode_1_max_min: dict[str, complex]
    _freq_mode_1_max_min: dict[str, complex]
    _lambda_mode_1_max_min: dict[str, complex]
    _Omega_rot_dim_mode_1_max_min: dict[str, float]
    _y_1_mode_1_max_min: dict[str, float]
    _y_2_mode_1_max_min: dict[str, float]
    _y_3_mode_1_max_min: dict[str, float]
    _y_4_mode_1_max_min: dict[str, float]
    _y_5_mode_1_max_min: dict[str, float] | None
    _y_6_mode_1_max_min: dict[str, float] | None
    _xi_r_mode_1_max_min: dict[str, float]
    _xi_h_mode_1_max_min: dict[str, float]
    _nabla_mode_1_max_min: dict[str, float]
    _lag_L_mode_1_max_min: dict[str, float]
    _eul_P_mode_1_max_min: dict[str, float]
    _eul_T_mode_1_max_min: dict[str, float]
    _lag_P_mode_1_max_min: dict[str, float]
    _lag_T_mode_1_max_min: dict[str, float]
    _eul_phi_mode_1_max_min: dict[str, float]
    _eul_rho_mode_1_max_min: dict[str, float]
    _lag_rho_mode_1_max_min: dict[str, float]
    _deul_phi_mode_1_max_min: dict[str, float]
    _nabla_ad_mode_1_max_min: dict[str, float]
    _xi_r_dim_mode_1_max_min: dict[str, float]
    _xi_h_dim_mode_1_max_min: dict[str, float]
    _lag_L_dim_mode_1_max_min: dict[str, float]
    _eul_P_dim_mode_1_max_min: dict[str, float]
    _eul_T_dim_mode_1_max_min: dict[str, float]
    _lag_P_dim_mode_1_max_min: dict[str, float]
    _lag_T_dim_mode_1_max_min: dict[str, float]
    _dnabla_ad_mode_1_max_min: dict[str, float]
    _eul_phi_dim_mode_1_max_min: dict[str, float]
    _eul_rho_dim_mode_1_max_min: dict[str, float]
    _lag_rho_dim_mode_1_max_min: dict[str, float]
    _deul_phi_dim_mode_1_max_min: dict[str, float]
    _omega_mode_2_max_min: dict[str, complex]
    _freq_mode_2_max_min: dict[str, complex]
    _lambda_mode_2_max_min: dict[str, complex]
    _Omega_rot_dim_mode_2_max_min: dict[str, float]
    _y_1_mode_2_max_min: dict[str, float]
    _y_2_mode_2_max_min: dict[str, float]
    _y_3_mode_2_max_min: dict[str, float]
    _y_4_mode_2_max_min: dict[str, float]
    _y_5_mode_2_max_min: dict[str, float] | None
    _y_6_mode_2_max_min: dict[str, float] | None
    _xi_r_mode_2_max_min: dict[str, float]
    _xi_h_mode_2_max_min: dict[str, float]
    _nabla_mode_2_max_min: dict[str, float]
    _lag_L_mode_2_max_min: dict[str, float]
    _eul_P_mode_2_max_min: dict[str, float]
    _eul_T_mode_2_max_min: dict[str, float]
    _lag_P_mode_2_max_min: dict[str, float]
    _lag_T_mode_2_max_min: dict[str, float]
    _eul_phi_mode_2_max_min: dict[str, float]
    _eul_rho_mode_2_max_min: dict[str, float]
    _lag_rho_mode_2_max_min: dict[str, float]
    _deul_phi_mode_2_max_min: dict[str, float]
    _nabla_ad_mode_2_max_min: dict[str, float]
    _xi_r_dim_mode_2_max_min: dict[str, float]
    _xi_h_dim_mode_2_max_min: dict[str, float]
    _lag_L_dim_mode_2_max_min: dict[str, float]
    _eul_P_dim_mode_2_max_min: dict[str, float]
    _eul_T_dim_mode_2_max_min: dict[str, float]
    _lag_P_dim_mode_2_max_min: dict[str, float]
    _lag_T_dim_mode_2_max_min: dict[str, float]
    _dnabla_ad_mode_2_max_min: dict[str, float]
    _eul_phi_dim_mode_2_max_min: dict[str, float]
    _eul_rho_dim_mode_2_max_min: dict[str, float]
    _lag_rho_dim_mode_2_max_min: dict[str, float]
    _deul_phi_dim_mode_2_max_min: dict[str, float]
    _omega_mode_3_max_min: dict[str, complex]
    _freq_mode_3_max_min: dict[str, complex]
    _lambda_mode_3_max_min: dict[str, complex]
    _Omega_rot_dim_mode_3_max_min: dict[str, float]
    _y_1_mode_3_max_min: dict[str, float]
    _y_2_mode_3_max_min: dict[str, float]
    _y_3_mode_3_max_min: dict[str, float]
    _y_4_mode_3_max_min: dict[str, float]
    _y_5_mode_3_max_min: dict[str, float] | None
    _y_6_mode_3_max_min: dict[str, float] | None
    _xi_r_mode_3_max_min: dict[str, float]
    _xi_h_mode_3_max_min: dict[str, float]
    _nabla_mode_3_max_min: dict[str, float]
    _lag_L_mode_3_max_min: dict[str, float]
    _eul_P_mode_3_max_min: dict[str, float]
    _eul_T_mode_3_max_min: dict[str, float]
    _lag_P_mode_3_max_min: dict[str, float]
    _lag_T_mode_3_max_min: dict[str, float]
    _eul_phi_mode_3_max_min: dict[str, float]
    _eul_rho_mode_3_max_min: dict[str, float]
    _lag_rho_mode_3_max_min: dict[str, float]
    _deul_phi_mode_3_max_min: dict[str, float]
    _nabla_ad_mode_3_max_min: dict[str, float]
    _xi_r_dim_mode_3_max_min: dict[str, float]
    _xi_h_dim_mode_3_max_min: dict[str, float]
    _lag_L_dim_mode_3_max_min: dict[str, float]
    _eul_P_dim_mode_3_max_min: dict[str, float]
    _eul_T_dim_mode_3_max_min: dict[str, float]
    _lag_P_dim_mode_3_max_min: dict[str, float]
    _lag_T_dim_mode_3_max_min: dict[str, float]
    _dnabla_ad_mode_3_max_min: dict[str, float]
    _eul_phi_dim_mode_3_max_min: dict[str, float]
    _eul_rho_dim_mode_3_max_min: dict[str, float]
    _lag_rho_dim_mode_3_max_min: dict[str, float]
    _deul_phi_dim_mode_3_max_min: dict[str, float]

    # superclass entity initialization
    def __init__(
        self,
        nr_modes: int,
        kwargs_diff_div: 'dict[str, typing.Any] | None' = None,
        kwargs_diff_terms: 'dict[str, typing.Any] | None' = None,
        diff_terms_method: str = 'gradient',
        polytropic: bool = False,
        analytic_polytrope: bool = False,
        cut_center: bool = True,
        cut_surface: bool = True,
        store_debug: bool = False,
    ) -> None:
        # store whether debug properties should be stored
        self._store_debug = store_debug
        # store the kwargs to be used in the differentiation methods
        # - used to compute divergences
        self._kwargs_diff_div = kwargs_diff_div
        # - used to compute the derivatives in the coupling terms
        self._kwargs_diff_terms = kwargs_diff_terms
        # store the differentiation method to be used in the different terms
        self._diff_terms_method = diff_terms_method
        # initialize the list of common attributes that only need to be loaded once
        self._list_common = ModeData.get_common_attrs(polytrope=polytropic)
        # store the utility attribute that denotes whether polytropic/GYRE or MESA/GYRE models are used
        self._use_polytropic = polytropic
        # store whether analytic prescriptions are used for polytropic data, computed based on solely polytrope data
        self._analytic_polytrope = analytic_polytrope
        # store the utility attribute that denotes how many modes are involved in the coupling coefficient computations
        self._nr_modes = nr_modes
        # -- NORMALIZATION FACTORS
        # initialize the normalization factors for the different terms
        self._norm_P = 1.0
        self._norm_rho = 1.0
        # -- CUT FACTOR
        # cut the center point of the profiles if True
        self._cut_center = cut_center
        # cut the surface point of the profiles if True
        self._cut_surface = cut_surface
        # -- DEBUG PROPERTIES
        # store the min_max debug property list
        if self._store_debug:
            self._attr_list_max_min = DmmScc.get_attrs()
        else:
            self._attr_list_max_min = []

    # GETTER/SETTER METHODS

    # INITIALIZATION HELPER METHODS

    # method that extracts information from the first mode information dictionary, which is common among all mode information dictionaries
    def _extract_common_info_dictionary(
        self, my_dict: 'dict[str, typing.Any]'
    ) -> None:
        """Internal initialization method used to initialize the information attributes shared among the different mode information dictionaries.

        TODO: rename to 'shared_info'

        Parameters
        ----------
        my_dict : dict
            The dictionary containing the attributes that need to be unpacked and stored in the object.
        """
        # create an exception list for the debugging properties:
        # these are numbers that could potentially be set to None (if not present in files)
        debug_exceptions = ['Delta_p', 'Delta_g']
        # loop over the common attributes and store them under the appropriate names
        for _attr in self._list_common:
            # retrieve the common attribute from the information dictionary,
            # if it is not present, store a None value
            setattr(self, f'_{_attr}', my_dict.get(_attr))
            # make a debug max/min attribute if appropriate
            if (_attr not in debug_exceptions) and self._store_debug:
                self._add_debug_attribute(my_attr=f'_{_attr}')

    # method that extracts common information from mode information dictionary
    def _extract_mode_dependent_info_dictionary(
        self, my_dict: 'dict[str, typing.Any]', mode_number: int
    ) -> None:
        """Internal initialization method used to initialize a bunch of mode-dependent parameters, based on the information stored in the information dictionary.

        Parameters
        ----------
        my_dict : dict[str, typing.Any]
            The information dictionary.
        mode_number : int
            The mode number that identifies for which mode you are storing information.
        """
        # loop over the keys and values of the data dictionary and store
        # them under the appropriate names
        for _k, _v in my_dict.items():
            # check that this is not a common attribute
            if _k not in self._list_common:
                # set the attribute for this class
                setattr(self, f'_{_k}_mode_{mode_number}', _v)
                # attempt to add a mode-dependent debug attribute
                if self._store_debug:
                    self._add_debug_attribute(
                        my_attr=f'_{_k}_mode_{mode_number}'
                    )

    # method that performs interpolations for mode dependent properties, if necessary
    def _interpolation_modes(
        self, list_dict: 'list[dict[str, typing.Any]]'
    ) -> None:
        """Internal method used to perform interpolations for mode dependent properties, if necessary.

        Parameters
        ----------
        list_dict : list[dict[str, typing.Any]]
            The list of parameter information dictionaries.
        """
        # store the interpolation list
        _interp_list = [
            'y_1',
            'y_2',
            'y_3',
            'y_4',
            'xi_r',
            'xi_h',
            'eul_phi',
            'deul_phi',
            'lag_L',
            'eul_P',
            'eul_rho',
            'eul_T',
            'lag_P',
            'lag_rho',
            'lag_T',
            'xi_r_dim',
            'xi_h_dim',
            'eul_phi_dim',
            'deul_phi_dim',
            'lag_L_dim',
            'eul_P_dim',
            'eul_rho_dim',
            'eul_T_dim',
            'lag_P_dim',
            'lag_rho_dim',
            'lag_T_dim',
            'y_5',
            'y_6',
        ]
        # check length of specific properties
        _lens = [_x['n'] for _x in list_dict]
        # get interpolation 'x'-coordinate of highest length
        _x_interpolation = list_dict[np.argmax(_lens)]['x']
        # perform interpolations, if necessary
        for _i, _my_dict in enumerate(list_dict):
            # interpolate using the dimensionless radial coordinate and
            # store interpolated profiles
            self._interpolator(_i + 1, _my_dict, _interp_list, _x_interpolation)
            # add debug properties
            if self._store_debug:
                self._add_debug_attribute_modes(_i + 1, _my_dict, _interp_list)

    def _add_debug_attribute(
        self, my_attr: str, normed: bool = False, poly: bool = False
    ) -> None:
        """Adds a debug property for a given mode attribute name.

        Parameters
        ----------
        my_attr : str
            Name of the mode attribute.
        normed : bool, optional
            If True, add a '_normed' suffix to the attribute name, if False, do not add anything to the attribute name; by default False.
        poly : bool, optional
            If True, add a '_poly' suffix to the attribute name, if False do not add anything to the attribute name. If 'normed' is also true, add a '_normed_poly' suffix to the attribute name; by default False.
        """
        # get the debug attribute name
        _dbg_attr_name = (
            f'{my_attr}_max_min_normed_poly'
            if (poly and normed)
            else f'{my_attr}_max_min_normed'
            if normed
            else f'{my_attr}_max_min_poly'
            if poly
            else f'{my_attr}_max_min'
        )
        # try to add attributes
        try:
            # get the value of the attribute
            _my_attr_val = getattr(self, my_attr)
            # set the attribute to a specific value (numpy array)
            setattr(
                self,
                _dbg_attr_name,
                {
                    'max': _my_attr_val.max(),
                    'min': _my_attr_val.min(),
                    'surface': _my_attr_val[-3:],
                    'core': _my_attr_val[:3],
                },
            )
        except IndexError:
            # this is supposed to be a numpy float/int, so we do not store anything
            pass
        except TypeError:
            # this object was not made to be sliced, so we do not store anything
            pass
        except AttributeError:
            try:
                # this is not a numpy array, so we do not store anything
                _my_attr_val = getattr(self, my_attr)
            except AttributeError:
                # this attribute simply does not exist!
                setattr(self, _dbg_attr_name, None)

    def _add_debug_attribute_modes(
        self,
        mode_nr: int,
        my_dict: 'dict[str, np.ndarray]',
        interp_list: list[str],
    ) -> None:
        """Adds the debug property for the interpolated mode attributes.

        Parameters
        ----------
        mode_nr : int
            The number of the mode for which debug attributes are stored.
        my_dict : dict[str, np.ndarray]
            Dictionary containing the data arrays of the specific mode for which interpolations might be necessary.
        interp_list : list
            Contains the names of the parameters for which interpolations were performed.
        """
        # store debug attributes for interpolated mode quantities
        for _k in my_dict.keys():
            # check if interpolation is needed
            if _k in interp_list:
                self._add_debug_attribute(my_attr=f'_{_k}_mode_{mode_nr}')

    def _interpolator(
        self,
        mode_nr: int,
        my_dict: 'dict[str, np.ndarray]',
        interp_list: list[str],
        rad_coordinates_max: 'np.ndarray',
    ):
        """Internal workhorse method for '_interpolation_modes'.

        Parameters
        ----------
        mode_nr : int
            The number of the mode for which interpolations are performed.
        my_dict : dict[str, np.ndarray]
            dictionary containing the data arrays of the specific mode for which interpolations might be necessary.
        interp_list : list[str]
            Contains the names of the parameters for which interpolation should be performed, if deemed necessary.
        rad_coordinates_max : np.ndarray
            The data array containing the radial coordinates for the mode that has the maximal number of data points.
        """
        # ensure that the values for the normalized radial coordinate corresponds to the max. nr. of bins for the interpolation
        self._x = rad_coordinates_max
        # perform interpolation for the necessary quantities
        for _k, _v in my_dict.items():
            # check if interpolation is needed
            if _k in interp_list:
                try:
                    logger.debug('Real array interpolation.')
                    # interpolate value
                    _interp = np.interp(self._x, my_dict['x'], _v)
                    # store the interpolated value
                    setattr(self, f'_{_k}_mode_{mode_nr}', _interp)
                except TypeError:
                    logger.debug('Complex array interpolation.')
                    # initialize the complex array
                    my_complex_array = re_im_parts(
                        real_val=np.interp(self._x, my_dict['x'], _v['re']),
                        imag_val=np.interp(self._x, my_dict['x'], _v['im']),
                        check_for_void=True,
                    )
                    # store the complex array
                    setattr(self, f'_{_k}_mode_{mode_nr}', my_complex_array)

    # method that extracts additional profile information
    def _extract_additional_profile_info(
        self, my_internal_structure_dict: 'dict[str, str | float | np.ndarray]'
    ) -> None:
        """Internal initialization method used to initialize a bunch of (structure) profile-specific parameters, based on the information stored in the information dictionary.

        Parameters
        ----------
        my_internal_structure_dict : dict[str, str | float | np.ndarray]
            Contains the additional MESA/Polytrope structure profile information.
        """
        # compute the normalized radial coordinate for interpolation purposes
        if self._use_polytropic:
            my_radius = my_internal_structure_dict['r']
            if typing.TYPE_CHECKING:
                assert isinstance(my_radius, np.ndarray)
            _norm_rad = my_radius / my_radius.max()
            # define an exclude list containing elements that should not be extracted
            _exclude_list = [
                'n',
                'n_r',
                'n_poly',
                'z',
                'theta',
                'dtheta',
                'drho',
                'drho2',
                'drho3',
                'dp',
                'dp2',
                'dp3',
                'dgravpot',
                'dgravpot2',
                'dgravpot3',
                'v',
                'surfgrav',
            ]
        else:
            my_radius = my_internal_structure_dict['radius']
            if typing.TYPE_CHECKING:
                assert isinstance(my_radius, np.ndarray)
            _norm_rad = (my_radius / my_radius.max())[::-1]
            # define an exclude list containing elements that should not be extracted
            _exclude_list = []
        # NOTE: have to reverse the MESA data arrays to fit GYRE convention
        # check if length is ok, otherwise perform interpolation
        if _norm_rad.shape[0] == self._x.shape[0]:
            logger.debug(
                'NO interpolations needed for POLYTROPIC STRUCTURE profiles.'
                if self._use_polytropic
                else 'NO interpolations needed for MESA profiles.'
            )
            # loop over the dictionary items
            for _k, _v in my_internal_structure_dict.items():
                # check if it should be extracted
                if _k not in _exclude_list:
                    # set the attribute and add profile suffix to parameter to distinguish it
                    if (
                        self._use_polytropic
                        or isinstance(_v, float)
                        or isinstance(_v, str)
                        or isinstance(_v, int)
                    ):
                        setattr(self, f'_{_k}_profile', _v)
                    else:
                        setattr(self, f'_{_k}_profile', _v[::-1])
                        # NOTE: have to reverse the MESA data arrays to fit GYRE convention
        else:
            logger.debug(
                'Interpolations needed for POLYTROPIC STRUCTURE profiles.'
                if self._use_polytropic
                else 'Interpolations needed for MESA profiles.'
            )
            # loop over the dictionary items
            for _k, _v in my_internal_structure_dict.items():
                # do interpolation, set the interpolated attribute and add profile suffix to parameter to distinguish it
                if (
                    isinstance(_v, float)
                    or isinstance(_v, str)
                    or isinstance(_v, int)
                ):
                    # no profile --> no interpolation
                    setattr(self, f'_{_k}', _v)
                else:
                    # profile --> needs interpolation
                    if self._use_polytropic:
                        _interp_v = np.interp(self._x, _norm_rad, _v)
                    else:
                        _interp_v = np.interp(self._x, _norm_rad, _v[::-1])
                        # NOTE: have to reverse the MESA data arrays to fit GYRE convention
                    setattr(self, f'_{_k}_profile', _interp_v)
            logger.debug(
                'Interpolations performed for PROFILE STRUCTURE profiles, where necessary.'
                if self._use_polytropic
                else 'Interpolations performed for MESA profiles, where necessary.'
            )

    # method that sets the normalization factors
    def _set_normalization_factors(
        self, P: 'np.ndarray | float', rho: 'np.ndarray | float'
    ) -> None:
        """Internal initialization method used to store/set a bunch of normalization factors, of different stellar/polytrope quantities.

        Parameters
        ----------
        P : float | np.ndarray
            The array of pressure values (MESA) or the P0 value (polytrope).
        rho : float | np.ndarray
            The array of density values (MESA) or the Rho0 value (polytrope).
        """
        # get numpy array slicer that will be used to slice arrays, if necessary
        match (self._cut_center, self._cut_surface):
            case (True, True):
                np_slicer = np.s_[1:-1]
            case (False, True):
                np_slicer = np.s_[:-1]
            case (True, False):
                np_slicer = np.s_[1:]
            case _:
                np_slicer = np.s_[:]
        # store the normalization factors
        self._norm_P = P[np_slicer].max() if isinstance(P, np.ndarray) else P
        self._norm_rho = (
            rho[np_slicer].max() if isinstance(rho, np.ndarray) else rho
        )

    # method that performs pre-profile generation cuts
    def _pre_profile_cuts(self) -> None:
        """Performs cuts of properties needed to compute/generate profiles."""
        # perform cuts of center points, if necessary
        if self._cut_center:
            # perform cuts prior to profile loading
            GYRECutter.cutter(
                self, 'pre_profile_load_cuts', polytrope=self._use_polytropic
            )
        # perform cuts of surface points, if necessary
        if self._cut_surface:
            # perform cuts prior to profile loading
            GYRECutter.cutter(
                self,
                'pre_profile_load_cuts',
                cut_slice=np.s_[:-1],
                polytrope=self._use_polytropic,
            )

    # method that computes the normed profiles used for MESA+GYRE data
    def _get_profiles_MESA_GYRE(
        self,
        kwargs_diff: 'dict[str, typing.Any]' = {},
        use_brunt_def: bool = True,
    ) -> None:
        """Internal initialization helper method that computes the normed profiles for specific data on the internal structure computed by MESA.

        Parameters
        ----------
        kwargs_diff : dict[str, typing.Any], optional
            Contains the key-word arguments for the numerical differentiator object; by default {}.
        use_brunt_def : bool, optional
            Decides whether you use a definition based on the Brunt-Väisälä or buoyancy frequency to define the gradient of the density, or if a  numerical estimate from the actual density profile is ok; by default True.
        """
        # - NOT NORMALIZED
        # perform cuts prior to profile loading
        self._pre_profile_cuts()
        # get the helper radial profile (and cut out center point, if necessary)
        radial = self._x * self._R_star
        # compute the derivative of the Newtonian gravitational potential = integrated Poisson's equation = dPhi/dr = (G * M_r) / (r^2)
        self._dgravpot = (self._G * self._M_r) / (radial ** (2.0))
        # compute the derivatives of the pressure, density and gravitational potential
        # - explicit numerical derivatives
        # initialize the numerical differentiator object suited for computing first-order derivatives
        _grad = NumericalDifferentiator(
            order_derivative=1, differentiation_method=self._diff_terms_method
        )
        # density derivatives
        if self._use_symbolic_derivative:
            # Use definition based on Brunt-Vaisala frequency (see e.g. Aerts et al., 2010; in RADIANS PER SECOND):
            # N^2 = g * (((1 / Gamma_1) * d lnP/dr) - dlnrho/dr)
            # where g = dgravpot, and we have used the substitution dP/dr = -rho * g
            # --> drho/dr = rho * [(-rho * g / (Gamma_1 * P)) - (N^2 / g)]
            if use_brunt_def:
                self._drho = self._rho * (
                    (
                        (-1.0 * self._rho * self._dgravpot)
                        / (self._Gamma_1 * self._P)
                    )
                    - (self._brunt_N2_profile / self._dgravpot)
                )
            else:
                _, self._drho = self._compute_der(
                    _grad, x=radial, y=self._rho, kwargs_diff=kwargs_diff
                )
        else:
            _, self._drho = self._compute_der(
                _grad, x=radial, y=self._rho, kwargs_diff=kwargs_diff
            )
        # compute the other derivatives numerically
        _, self._drho2 = self._compute_der(
            _grad, x=radial, y=self._drho, kwargs_diff=kwargs_diff
        )
        _, self._drho3 = self._compute_der(
            _grad, x=radial, y=self._drho2, kwargs_diff=kwargs_diff
        )
        # store debug data if necessary:
        if self._store_debug:
            # store the maxima and minima of the density profile
            self._add_debug_attribute('_rho')
            # store the maxima and minima of the density derivatives
            self._add_debug_attribute('_drho')
            self._add_debug_attribute('_drho2')
            self._add_debug_attribute('_drho3')
        # gravitational potential derivatives
        if self._use_symbolic_derivative:
            # Poisson's equation: d^2 Phi / dr^2 = 4 * Pi * G * rho - (2/r) * dPhi/dr
            self._dgravpot2 = (4.0 * np.pi * self._G * self._rho) - (
                self._dgravpot * (2.0 / radial)
            )
            # derivative of Poisson's equation: d^3 Phi / dr^3 = 4 * Pi * G * drho/dr
            # = 4.0 * np.pi * self._G * self._drho
            self._dgravpot3 = (4.0 * np.pi * self._G * self._drho) + (
                2.0 / radial
            ) * ((self._dgravpot / radial) - self._dgravpot2)
        else:
            # numerical computation of the derivatives
            _, self._dgravpot2 = self._compute_der(
                _grad, x=radial, y=self._dgravpot, kwargs_diff=kwargs_diff
            )
            _, self._dgravpot3 = self._compute_der(
                _grad, x=radial, y=self._dgravpot2, kwargs_diff=kwargs_diff
            )
        # store debug data if necessary:
        if self._store_debug:
            # store the maxima and minima of the gravitational potential derivatives
            self._add_debug_attribute('_dgravpot')
            self._add_debug_attribute('_dgravpot2')
            self._add_debug_attribute('_dgravpot3')
        # pressure derivatives
        if self._use_symbolic_derivative:
            # based on hydrostatic support/equilibrium(!)
            # --> dP/dr = - G * m(r) * rho / r^2 = - dPhi/dr * rho (assuming spherical symmetry of hydrostatic equilibrium state)
            # self._dP = - self._G * self._M_r * self._rho / (radial**(2.0)) = -self._dgravpot * self._rho
            self._dP = -self._dgravpot * self._rho
            # additional derivatives:
            # --> d^2P/dr^2 = - G * rho^2 * 4 * Pi - G * m(r) * drho(r) / r^2 - (2/r) * dP = -d^2Phi/dr^2 * rho - dPhi/dr * drho/dr
            # --> d^3P/dr^3 = - 12 * pi * G * rho * drho(r) - G * m(r) * drho2(r) / r^2 + 2 * G * m(r) * drho(r) / r^3 + 2*dP/r^2 - (2/r)*dP2 = -d^3Phi/dr^3 * rho - 2 * d^2Phi/dr^2 * drho/dr - dPhi/dr * d^2rho/dr^2
            self._dP2 = (
                -self._dgravpot2 * self._rho - self._dgravpot * self._drho
            )
            self._dP3 = (
                -self._dgravpot3 * self._rho
                - 2.0 * self._dgravpot2 * self._drho
                - self._dgravpot * self._drho2
            )
        else:
            # numerical computation of derivative
            _, self._dP = self._compute_der(
                _grad, x=radial, y=self._P, kwargs_diff=kwargs_diff
            )
            _, self._dP2 = self._compute_der(
                _grad, x=radial, y=self._dP, kwargs_diff=kwargs_diff
            )
            _, self._dP3 = self._compute_der(
                _grad, x=radial, y=self._dP2, kwargs_diff=kwargs_diff
            )
        # store debug data, if necessary:
        if self._store_debug:
            # store the maxima and minima of the pressure profile
            self._add_debug_attribute('_P')
            # store the maxima and minima of the pressure derivatives
            self._add_debug_attribute('_dP')
            self._add_debug_attribute('_dP2')
            self._add_debug_attribute('_dP3')
        # - NORMALIZED
        # compute the normalized pressure and density profiles
        # - pressure
        self._P_normed = self._P / self._norm_P
        self._dP_normed = self._dP / self._norm_P
        self._dP2_normed = self._dP2 / self._norm_P
        self._dP3_normed = self._dP3 / self._norm_P
        # - density
        self._rho_normed = self._rho / self._norm_rho
        self._drho_normed = self._drho / self._norm_rho
        self._drho2_normed = self._drho2 / self._norm_rho
        self._drho3_normed = self._drho3 / self._norm_rho
        # -- no need to compute the normalized profiles of the gravitational potential, because we use non-normalized profiles in the computations!
        # store debug data if necessary:
        if self._store_debug:
            # store the maxima and minima of the pressure and density profiles
            self._add_debug_attribute('_P', normed=True)
            self._add_debug_attribute('_rho', normed=True)
            # store the maxima and minima of the normalized derivatives
            self._add_debug_attribute('_drho', normed=True)
            self._add_debug_attribute('_drho2', normed=True)
            self._add_debug_attribute('_drho3', normed=True)
            self._add_debug_attribute('_dP', normed=True)
            self._add_debug_attribute('_dP2', normed=True)
            self._add_debug_attribute('_dP3', normed=True)
            # fill debug attributes that have not been initialized
            self._fill_debug_attributes(self._attr_list_max_min)

    # method that computes the normed profiles used for polytrope data, emulating the MESA+GYRE approach
    def _get_profiles_polytrope_MESA_GYRE(
        self,
        kwargs_diff: 'dict[str, typing.Any]' = {},
        use_brunt_def: bool = True,
    ) -> None:
        """Internal initialization helper method that computes the normed profiles for specific data on the internal structure computed for the GYRE polytrope.

        Parameters
        ----------
        kwargs_diff : dict, optional
            Contains the key-word arguments for the numerical differentiator object; by default {}.
        use_brunt_def : bool, optional
            Decides whether you use a definition based on the Brunt-Väisälä or buoyancy frequency to define the gradient of the density, or if a  numerical estimate from the actual density profile is ok; by default True.
        """
        # - NOT NORMALIZED
        # perform cuts prior to profile loading
        self._pre_profile_cuts()
        # compute the derivative of the Newtonian gravitational potential = integrated Poisson's equation = dPhi/dr = (G * M_r) / (r^2)
        self._dgravpot = (self._G_profile * self._mr_profile) / (
            self._r_profile ** (2.0)
        )
        # initialize the numerical differentiator object suited for computing first-order derivatives
        _grad = NumericalDifferentiator(
            order_derivative=1, differentiation_method=self._diff_terms_method
        )
        # density derivatives
        if self._use_symbolic_derivative:
            # Use definition based on Brunt-Vaisala frequency (see e.g. Aerts et al., 2010; IN RADIANS PER SECOND):
            # N^2 = g * (((1 / Gamma_1) * d lnP/dr) - dlnrho/dr)
            # where g = dgravpot, and we have used the substitution dP/dr = -rho * g
            # --> drho/dr = rho * [(-rho * g / (Gamma_1 * P)) - (N^2 / g)]
            if use_brunt_def:
                self._drho = self._rho_profile * (
                    (
                        (-1.0 * self._rho_profile * self._dgravpot)
                        / (self._Gamma_1 * self._p_profile)
                    )
                    - (self._brunt_n2_profile / self._dgravpot)
                )
            else:
                _, self._drho = self._compute_der(
                    _grad,
                    x=self._r_profile,
                    y=self._rho,
                    kwargs_diff=kwargs_diff,
                )
        else:
            _, self._drho = self._compute_der(
                _grad,
                x=self._r_profile,
                y=self._rho_profile,
                kwargs_diff=kwargs_diff,
            )
        # compute the other derivatives numerically
        _, self._drho2 = self._compute_der(
            _grad, x=self._r_profile, y=self._drho, kwargs_diff=kwargs_diff
        )
        _, self._drho3 = self._compute_der(
            _grad, x=self._r_profile, y=self._drho2, kwargs_diff=kwargs_diff
        )
        # store debug data if necessary:
        if self._store_debug:
            # store the maxima and minima of the density profile
            self._add_debug_attribute('_rho_profile')
            # store the maxima and minima of the density derivatives
            self._add_debug_attribute('_drho')
            self._add_debug_attribute('_drho2')
            self._add_debug_attribute('_drho3')
        # gravitational potential derivatives
        if self._use_symbolic_derivative:
            # Poisson's equation: d^2 Phi / dr^2 = 4 * Pi * G * rho - (2/r) * dPhi/dr
            self._dgravpot2 = (
                4.0 * np.pi * self._G_profile * self._rho_profile
            ) - (self._dgravpot * (2.0 / self._r_profile))
            # derivative of Poisson's equation: d^3 Phi / dr^3 = 4 * Pi * G * drho/dr = 4.0 * np.pi * self._G * self._drho
            self._dgravpot3 = (4.0 * np.pi * self._G * self._drho) + (
                2.0 / self._r_profile
            ) * ((self._dgravpot / self._r_profile) - self._dgravpot2)
        else:
            # numerical computation of the derivatives
            _, self._dgravpot2 = self._compute_der(
                _grad,
                x=self._r_profile,
                y=self._dgravpot,
                kwargs_diff=kwargs_diff,
            )
            _, self._dgravpot3 = self._compute_der(
                _grad,
                x=self._r_profile,
                y=self._dgravpot2,
                kwargs_diff=kwargs_diff,
            )
        # store debug data if necessary:
        if self._store_debug:
            # store the maxima and minima of the gravitational potential derivatives
            self._add_debug_attribute('_dgravpot')
            self._add_debug_attribute('_dgravpot2')
            self._add_debug_attribute('_dgravpot3')
        # pressure derivatives
        if self._use_symbolic_derivative:
            # based on hydrostatic support/equilibrium(!)
            # --> dP/dr = - G * m(r) * rho / r^2 = - dPhi/dr * rho (assuming spherical symmetry of hydrostatic equilibrium state)
            # self._dP = - self._G * self._M_r * self._rho / (radial**(2.0)) = -self._dgravpot * self._rho
            self._dP = -self._dgravpot * self._rho_profile
            # additional derivatives:
            # --> d^2P/dr^2 = - G * rho^2 * 4 * Pi - G * m(r) * drho(r) / r^2 - (2/r) * dP = -d^2Phi/dr^2 * rho - dPhi/dr * drho/dr
            # --> d^3P/dr^3 = - 12 * pi * G * rho * drho(r) - G * m(r) * drho2(r) / r^2 + 2 * G * m(r) * drho(r) / r^3 + 2*dP/r^2 - (2/r)*dP2 = -d^3Phi/dr^3 * rho - 2 * d^2Phi/dr^2 * drho/dr - dPhi/dr * d^2rho/dr^2
            self._dP2 = (
                -self._dgravpot2 * self._rho_profile
                - self._dgravpot * self._drho
            )
            self._dP3 = (
                -self._dgravpot3 * self._rho_profile
                - 2.0 * self._dgravpot2 * self._drho
                - self._dgravpot * self._drho2
            )
        else:
            # numerical computation of derivative
            _, self._dP = self._compute_der(
                _grad,
                x=self._r_profile,
                y=self._p_profile,
                kwargs_diff=kwargs_diff,
            )
            _, self._dP2 = self._compute_der(
                _grad, x=self._r_profile, y=self._dP, kwargs_diff=kwargs_diff
            )
            _, self._dP3 = self._compute_der(
                _grad, x=self._r_profile, y=self._dP2, kwargs_diff=kwargs_diff
            )
        # store debug data, if necessary:
        if self._store_debug:
            # store the maxima and minima of the pressure profile
            self._add_debug_attribute('_p_profile')
            # store the maxima and minima of the pressure derivatives
            self._add_debug_attribute('_dP')
            self._add_debug_attribute('_dP2')
            self._add_debug_attribute('_dP3')
        # - NORMALIZED
        # compute the normalized pressure and density profiles
        # - pressure
        self._P_normed = self._p_profile / self._norm_P
        self._dP_normed = self._dP / self._norm_P
        self._dP2_normed = self._dP2 / self._norm_P
        self._dP3_normed = self._dP3 / self._norm_P
        # - density
        self._rho_normed = self._rho_profile / self._norm_rho
        self._drho_normed = self._drho / self._norm_rho
        self._drho2_normed = self._drho2 / self._norm_rho
        self._drho3_normed = self._drho3 / self._norm_rho
        # -- no need to compute the normalized profiles of the gravitational potential, because we use non-normalized profiles in the computations!
        # store debug data if necessary:
        if self._store_debug:
            # store the maxima and minima of the pressure and density profiles
            self._add_debug_attribute('_p_profile', normed=True)
            self._add_debug_attribute('_rho_profile', normed=True)
            # store the maxima and minima of the normalized derivatives
            self._add_debug_attribute('_drho', normed=True)
            self._add_debug_attribute('_drho2', normed=True)
            self._add_debug_attribute('_drho3', normed=True)
            self._add_debug_attribute('_dP', normed=True)
            self._add_debug_attribute('_dP2', normed=True)
            self._add_debug_attribute('_dP3', normed=True)
            # fill debug attributes that have not been initialized
            self._fill_debug_attributes(self._attr_list_max_min)

    # get and/or interpolate
    def _get_and_interpolate_if_needed(
        self,
        my_attr: str,
        interp_necessary: bool,
        my_dict: 'dict[str, typing.Any]',
        radial_structure_profile: 'np.ndarray',
    ) -> 'np.ndarray | None':
        """Helper method used to retrieve structure info based on polytropic quantities.

        Parameters
        ----------
        my_attr : str
            The specific attribute of which you are retrieving the value.
        interp_necessary : bool
            Whether interpolation is necessary to fit to a GYRE grid.
        my_dict : dict[str, typing.Any]
            The dictionary containing the attributes that need to be unpacked and stored in the object.
        radial_structure_profile : np.ndarray
            Normalized radius helper array used for interpolation, if necessary.

        Returns
        -------
        np.ndarray | None
            The requested attribute from 'my_dict', possibly interpolated to fit to a GYRE grid.
        """
        if interp_necessary:
            # get the attribute
            _my_att = my_dict.get(my_attr)
            # try interpolation to return GYRE-fitted attribute
            if typing.TYPE_CHECKING:
                assert isinstance(_my_att, np.ndarray)
            if np.isnan(
                my_interp := np.interp(
                    self._x, radial_structure_profile, _my_att
                )
            ):  # if it is NaN, _my_att is None!
                return None
            else:
                return my_interp
        else:
            # no interpolation necessary: directly return attribute
            my_att = my_dict.get(my_attr)
            if my_att is None:
                return None
            else:
                if typing.TYPE_CHECKING:
                    assert isinstance(my_att, np.ndarray)
                return my_att

    # method that computes the normed profiles used for polytrope data, based on analytic expressions using polytrope data
    def _get_profiles_polytrope_analytic(
        self, my_dict: dict[str, typing.Any]
    ) -> None:
        """Internal initialization helper method that retrieves the normed profiles based on the internal structure computed for the GYRE polytrope.

        Parameters
        ----------
        my_dict : dict
            The dictionary containing the attributes that need to be unpacked and stored in the object.
        """
        # compute the helper normalized radius model grid for possibly necessary interpolations
        _help_r = self._r_profile / self._R_star
        # check if interpolations are needed
        _interp_needed = not (_help_r.shape[0] == self._x.shape[0])
        # store common kwargs for interpolation
        _common_kwargs = {
            'my_dict': my_dict,
            'interp_necessary': _interp_needed,
            'radial_structure_profile': _help_r,
        }
        # - NOT NORMALIZED
        # load the structure profile derivatives based on polytropic quantities / Newtonian gravitational potential derivatives
        self._dgravpot = self._get_and_interpolate_if_needed(
            my_attr='dgravpot', **_common_kwargs
        )
        self._dgravpot2 = self._get_and_interpolate_if_needed(
            my_attr='dgravpot2', **_common_kwargs
        )
        self._dgravpot3 = self._get_and_interpolate_if_needed(
            my_attr='dgravpot3', **_common_kwargs
        )
        # density derivatives
        self._drho = self._get_and_interpolate_if_needed(
            my_attr='drho', **_common_kwargs
        )
        self._drho2 = self._get_and_interpolate_if_needed(
            my_attr='drho2', **_common_kwargs
        )
        self._drho3 = self._get_and_interpolate_if_needed(
            my_attr='drho3', **_common_kwargs
        )
        # pressure derivatives
        self._dP = self._get_and_interpolate_if_needed(
            my_attr='dp', **_common_kwargs
        )
        self._dP2 = self._get_and_interpolate_if_needed(
            my_attr='dp2', **_common_kwargs
        )
        self._dP3 = self._get_and_interpolate_if_needed(
            my_attr='dp3', **_common_kwargs
        )
        # perform the cuts
        if self._cut_center:  # center point
            GYRECutter.cutter(
                self, 'profile_load_cuts_analytic', polytrope=True
            )
        if self._cut_surface:  # surface point
            GYRECutter.cutter(
                self,
                'profile_load_cuts_analytic',
                polytrope=True,
                cut_slice=np.s_[:-1],
            )
        # store the debug quantities, if necessary
        if self._store_debug:
            # store the maxima and minima of the density profile
            self._add_debug_attribute('_rho_profile', poly=True)
            # store the maxima and minima of the density derivatives
            self._add_debug_attribute('_drho', poly=True)
            self._add_debug_attribute('_drho2', poly=True)
            self._add_debug_attribute('_drho3', poly=True)
            # store the maxima and minima of
            # the gravitational potential derivatives
            self._add_debug_attribute('_dgravpot', poly=True)
            self._add_debug_attribute('_dgravpot2', poly=True)
            self._add_debug_attribute('_dgravpot3', poly=True)
            # store the maxima and minima of the pressure profile
            self._add_debug_attribute('_p_profile', poly=True)
            # store the maxima and minima of
            # the pressure derivatives
            self._add_debug_attribute('_dP', poly=True)
            self._add_debug_attribute('_dP2', poly=True)
            self._add_debug_attribute('_dP3', poly=True)
        # - NORMALIZED
        # compute the normalized pressure and density profiles
        # - pressure
        self._P_normed = self._p_profile / self._norm_P
        self._dP_normed = None if self._dP is None else self._dP / self._norm_P
        self._dP2_normed = (
            None if self._dP2 is None else self._dP2 / self._norm_P
        )
        self._dP3_normed = (
            None if self._dP3 is None else self._dP3 / self._norm_P
        )
        # - density
        self._rho_normed = self._rho_profile / self._norm_rho
        self._drho_normed = (
            None if self._drho is None else self._drho / self._norm_rho
        )
        self._drho2_normed = (
            None if self._drho2 is None else self._drho2 / self._norm_rho
        )
        self._drho3_normed = (
            None if self._drho3 is None else self._drho3 / self._norm_rho
        )
        # -- no need to compute the normalized profiles of the gravitational potential, because we use non-normalized profiles in the computations!
        # store debug data if necessary:
        if self._store_debug:
            # store the maxima and minima of the pressure and density profiles
            self._add_debug_attribute('_p_profile', poly=True, normed=True)
            self._add_debug_attribute('_rho_profile', poly=True, normed=True)
            # store the maxima and minima of the normalized derivatives
            self._add_debug_attribute('_drho', poly=True, normed=True)
            self._add_debug_attribute('_drho2', poly=True, normed=True)
            self._add_debug_attribute('_drho3', poly=True, normed=True)
            self._add_debug_attribute('_dP', poly=True, normed=True)
            self._add_debug_attribute('_dP2', poly=True, normed=True)
            self._add_debug_attribute('_dP3', poly=True, normed=True)
            # fill debug attributes that have not been initialized
            self._fill_debug_attributes(self._attr_list_max_min)

    # utility method that computes derivatives of a specific function (array of samples)
    def _compute_der(
        self,
        differentiator: NumericalDifferentiator,
        x: 'np.ndarray',
        y: 'np.ndarray',
        kwargs_diff: 'dict[str, typing.Any] | None',
        print_statement: bool = True,
    ) -> list[np.ndarray]:
        """Internal method used to compute a numerical derivative.

        Parameters
        ----------
        differentiator : NumericalDifferentiatior
            The initialized NumericalDifferentiator object used to obtain the numerical derivative.
        x : np.ndarray
            The numpy array of x-values.
        y : np.ndarray
            The numpy array of y-values.
        kwargs_diff : dict | None
            If None, the default keyword argument for the numerical differentiation action are used. If a dictionary is passed, its keyword arguments are passed on to the numerical differentiator.
        print_statement : bool, optional
            If True, and verbose output is selected, print message. If False, do not print message; by default True.

        Returns
        -------
        num_der_list: list[np.ndarray]
            The list containing numerical derivatives up to a certain order specified in the 'differentiator'.
        """
        # check if the array has a complex dtype
        if isinstance(y, np.ndarray) and ('complex' in str(y.dtype)):
            # compute the real part (in a nested way)
            _num_der_list_real = self._compute_der(
                differentiator=differentiator,
                x=x,
                y=y.real,
                kwargs_diff=kwargs_diff,
                print_statement=False,
            )
            # check if you want to compute complex derivatives
            if self._use_complex:
                # compute the complex part (in a nested way)
                _num_der_list_imag = self._compute_der(
                    differentiator=differentiator,
                    x=x,
                    y=y.imag,
                    kwargs_diff=kwargs_diff,
                    print_statement=False,
                )
                # construct the complex array output list
                num_der_list = [
                    _n_r if _n_r is None else re_im_parts(_n_r, _n_i)
                    for _n_r, _n_i in zip(
                        _num_der_list_real, _num_der_list_imag
                    )
                ]
            else:
                # only the real part is needed (copy by value)
                num_der_list = copy.deepcopy(_num_der_list_real)
        elif isinstance(y, np.ndarray):
            # compute the numerical derivative, with actions depending on the value of 'kwargs_diff'
            if kwargs_diff is None:
                num_der_list, _num_der_string = differentiator.differentiate(
                    x, y
                )
            else:
                num_der_list, _num_der_string = differentiator.differentiate(
                    x, y, **kwargs_diff
                )
        else:
            logger.error(
                'Numerical derivation not implemented for other data types! Now exiting.'
            )
            sys.exit()
        # log information if necessary (i.e. if verbose printing is enabled, and the logical "print_statement" input variable is True)
        if print_statement:
            logger.debug(_num_der_string)  # type: ignore
        # return the list containing numerical derivatives
        return num_der_list

    # method that stores the polytropic structure coefficients
    def _store_polytropic_structure_coefficients(self) -> None:
        """Internal initialization helper method that retrieves the structure coefficients based on the internal structure computed for the GYRE polytrope."""
        # LOAD STRUCTURE COEFFICIENTS
        # load c1
        self._c_1 = self._c1_profile.copy()
        # load V_2
        self._V_2 = self._v2_profile.copy()
        # load As
        self._As = self._as_profile.copy()
        # load U
        self._U = self._u_profile.copy()
        # STORE DEBUG ATTRIBUTES IF NECESSARY
        if self._store_debug:
            self._add_debug_attribute('_c_1', poly=True)
            self._add_debug_attribute('_V_2', poly=True)
            self._add_debug_attribute('_As', poly=True)
            self._add_debug_attribute('_U', poly=True)
            self._add_debug_attribute('_mr_profile', poly=True)

    # method that cuts out the innermost and outermost point of the quantities of interest, if requested: neglect the very center and/or the very surface
    def _mode_structure_cuts(self) -> None:
        """Internal utility method used to cut out the center and/or surface points of the mode/structure quantities of interest for the computation of coupling coefficients."""
        # cut specific attributes for polytrope or GYRE+MESA models
        if self._use_polytropic:
            # cut polytrope model attributes
            if self._cut_center:
                GYRECutter.cutter(self, 'polytropic_cuts')
                GYRECutter.cutter(self, 'structure_coefficient_cuts')
            if self._cut_surface:
                GYRECutter.cutter(self, 'polytropic_cuts', cut_slice=np.s_[:-1])
                GYRECutter.cutter(
                    self, 'structure_coefficient_cuts', cut_slice=np.s_[:-1]
                )
        else:
            # cut GYRE+MESA model attributes
            if self._cut_center:  # center point
                GYRECutter.cutter(self, 'mesa_gyre_cuts')
                GYRECutter.cutter(self, 'structure_coefficient_cuts')
            if self._cut_surface:  # surface point
                GYRECutter.cutter(self, 'mesa_gyre_cuts', cut_slice=np.s_[:-1])
                GYRECutter.cutter(
                    self, 'structure_coefficient_cuts', cut_slice=np.s_[:-1]
                )

    # method used to fill up the debug properties that were not yet initialized
    def _fill_debug_attributes(self, list_attrs: list[str]) -> None:
        """Gives uninitialized debug properties the None value.

        Parameters
        ----------
        list_attrs: list[str]
            Contains the attribute names of the attributes for which debug properties need to be stored.
        """
        # loop through the list and access the properties
        for _dbg_prop in list_attrs:
            # try to access the property
            try:
                getattr(self, _dbg_prop)
            except AttributeError:
                setattr(self, _dbg_prop, None)
