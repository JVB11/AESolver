"""Python module file containing the QuadraticCouplingCoefficient class that handles the computation of quadratic coupling coefficients.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import logging
import sys
import numpy as np
from collections import defaultdict
from itertools import permutations
from multimethod import multimethod
from tqdm import tqdm


# import intra-package modules
# - superclass
from .coefficient_support_classes import CCR

# - listing module passed to eigenfunction normalizer object
from .enumeration_files import EnumRadial, EnumRadialPoly

# - Hough function computer + integrating module
from ..angular_integration import HFHandler, HI

# - FreqHandler for typing
from ..frequency_handling import FH

# import debugging module enumeration
from ..coupling_coefficients.enumeration_files.enumerated_debug_properties import (
    DebugMinMaxQuadraticCouplingCoefficientRotating as DmmQccR,
)

# import custom packages
# - Numerical integration module
from num_integrator import cni  # type: ignore

# - Estimation of LTE eigenvalues using GYRE submodule
from lam_tar_gyre import GyreLambdas

# Eigenfunction normalizer
from gyre_eigen_norm import GYREEigenFunctionNormalizer as GYRENorm
from hough_function import HNorm  # Hough function normalizing module

# type checking imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # type checking types
    from typing import Any, Protocol, Self

    class RadIntProt(Protocol):
        def __call__(
            self: Self,
            integrand: np.ndarray,
            integrating_quantity: np.ndarray,
            indexer: slice = np.s_[:],
        ) -> float | np.ndarray:
            ...


# set up logger
logger = logging.getLogger(__name__)


# ---------------- QUADRATIC COUPLING CLASS (Rotating) ----------------
class QuadraticCouplingCoefficientRotating(CCR):
    """Python subclass that implements the specific functionalities for computing quadratic coupling coefficients in rotating stars.

    Parameters
    ----------
    first_dict : dict
        The dictionary containing the necessary mode information for the first mode of the triplet (mode A).
    second_dict : dict
        The dictionary containing the necessary mode information for the second mode of the triplet (mode B).
    third_dict : dict
        The dictionary containing the necessary mode information for the third mode of the triplet (mode C).
    additional_profile_information : dict
        The dictionary containing additional information on the stellar evolution code (MESA/POLYTROPE) profile output.
    kwargs_diff_div : dict or None, optional
        The keyword arguments dictionary that sets arguments to be used for computing divergences and their derivatives. If None, no keyword arguments are passed; by default None.
    kwargs_diff_terms : dict or None, optional
        The keyword arguments dictionary that sets arguments to be used for computing derivatives of terms in the coupling coefficient. If None, no keyword arguments are passed; by default None.
    diff_terms_method : str, optional
        Denotes the numerical method used to compute (numerical) derivatives for the terms in the coupling coefficient, if necessary; by default 'gradient'.
    use_complex : bool, optional
        If True, attempt to compute coupling coefficient using complex quantities. If False, only use real quantities; by default False. NOTE: this is not implemented, because the formalism assumes non-complex eigenfunctions and eigenvalues...
    polytropic : bool, optional
        If True, perform the necessary actions to compute the coupling coefficient for polytrope input. If False, assume stellar evolution model input for the computations; by default False.
    polytrope_data_dict : dict or None, optional
        Dictionary containing additional information on the polytrope model to be used to set up computations using polytrope input. If this is None and 'polytropic' is True, an error will occur. Ignored if 'polytropic' is False; by default None.
    self_coupling : bool, optional
        If True, use symbolic expressions that reduce the number of computations/computational load, in the assumption that mode 1 is coupling to itself. If False, use the general expressions; by default False. NOTE: self-coupling does not exist in a quadratic formalism!
    use_brunt_def : bool, optional
        If True and 'use_symbolic_derivatives' is True, use a symbolic expression in terms of the buoyancy/Brunt-Väisälä frequency to compute the first radial mass density derivative. If False, the first radial mass density derivative is computed numerically, unless polytropic models are used and 'analytic_polytrope' is True; by default True.
    use_symbolic_derivatives : bool, optional
        If True, use symbolic expressions to compute most of the stellar evolution model derivatives (except for radial derivatives of the mass density of order > 1). If False, compute these derivatives numerically; by default True.
    conj_cc : list[bool] | None, optional
        Determines which of the modes in the expression for the coupling coefficient are complex-conjugated (True) or not (False). If None, use the default value of [True, False, False]; by default None, which translates into [True, False, False]. (i.e. the coupling coefficient defined in Van Beeck et al. (2023))
    store_debug : bool, optional
        If True, store debug properties/attributes. If False, do not store these attributes; by default False.
    """

    # attribute type declaration
    _use_complex: bool
    _conj_cc: bool | tuple[bool, bool, bool]
    _freq_handler: FH
    _omp_threads: int
    _use_parallel: bool
    _numerical_integration_method: str
    _use_cheby: bool
    _cheby_order_mul: int
    _cheby_blow_up: bool
    _cheby_blow_up_fac: float
    _use_symbolic_derivative: bool
    _radint: 'RadIntProt'  # 'Callable[[np.ndarray, np.ndarray, Any], float]'
    _M_star: float
    _R_star: float
    _G: 'float | np.ndarray'
    _l_list: list[int]
    _m_list: list[int]
    _nu_list: list[float]
    _lambda_list: list[float]
    _npts: int
    _self_coupling: bool
    _normalizing_factors_modes: 'np.ndarray'
    _cc_term_1: None | float | complex
    _cc_term_2: None | float | complex
    _cc_term_3: None | float | complex
    _cc_term_4: None | float | complex
    _coupling_coefficient: None | float | complex
    _coupling_coefficient_integrated: 'None | np.ndarray'
    _cc_term_1_integrated: 'np.ndarray'
    _cc_term_2_integrated: 'np.ndarray'
    _cc_term_3_integrated: 'np.ndarray'
    _cc_term_4_integrated: 'np.ndarray'
    # mu value attributes, TODO: add theta values
    # for div_sin integrals
    _mu_values: 'np.ndarray'
    _mu_values_min_max: 'np.ndarray'  # debug attribute
    _theta_values: 'np.ndarray'
    _theta_values_min_max: 'np.ndarray'
    # mode-specific attributes
    _hr_mode_1: 'np.ndarray'
    _ht_mode_1: 'np.ndarray'
    _hp_mode_1: 'np.ndarray'
    _theta_der_hr_mode_1: 'np.ndarray'
    _theta_der_ht_mode_1: 'np.ndarray'
    _theta_der_hp_mode_1: 'np.ndarray'
    _theta_phi_mode_1: 'np.ndarray'
    _phi_phi_mode_1: 'np.ndarray'
    _mhr_mode_1: 'np.ndarray'
    _theta_der2_hr_mode_1: 'np.ndarray'
    _theta_der_hr_theta_mode_1: 'np.ndarray'
    _theta_der2_hr_theta_mode_1: 'np.ndarray'
    _theta_der_ht_theta_mode_1: 'np.ndarray'
    _theta_der_hp_theta_mode_1: 'np.ndarray'
    _hr_mode_2: 'np.ndarray'
    _ht_mode_2: 'np.ndarray'
    _hp_mode_2: 'np.ndarray'
    _theta_der_hr_mode_2: 'np.ndarray'
    _theta_der_ht_mode_2: 'np.ndarray'
    _theta_der_hp_mode_2: 'np.ndarray'
    _theta_phi_mode_2: 'np.ndarray'
    _phi_phi_mode_2: 'np.ndarray'
    _mhr_mode_2: 'np.ndarray'
    _theta_der2_hr_mode_2: 'np.ndarray'
    _theta_der_hr_theta_mode_2: 'np.ndarray'
    _theta_der2_hr_theta_mode_2: 'np.ndarray'
    _theta_der_ht_theta_mode_2: 'np.ndarray'
    _theta_der_hp_theta_mode_2: 'np.ndarray'
    _hr_mode_3: 'np.ndarray'
    _ht_mode_3: 'np.ndarray'
    _hp_mode_3: 'np.ndarray'
    _theta_der_hr_mode_3: 'np.ndarray'
    _theta_der_ht_mode_3: 'np.ndarray'
    _theta_der_hp_mode_3: 'np.ndarray'
    _theta_phi_mode_3: 'np.ndarray'
    _phi_phi_mode_3: 'np.ndarray'
    _mhr_mode_3: 'np.ndarray'
    _theta_der2_hr_mode_3: 'np.ndarray'
    _theta_der_hr_theta_mode_3: 'np.ndarray'
    _theta_der2_hr_theta_mode_3: 'np.ndarray'
    _theta_der_ht_theta_mode_3: 'np.ndarray'
    _theta_der_hp_theta_mode_3: 'np.ndarray'
    # mode-specific debug attributes
    _hr_mode_1_max_min: 'np.ndarray'
    _ht_mode_1_max_min: 'np.ndarray'
    _hp_mode_1_max_min: 'np.ndarray'
    _theta_der_hr_mode_1_max_min: 'np.ndarray'
    _theta_der_ht_mode_1_max_min: 'np.ndarray'
    _theta_der_hp_mode_1_max_min: 'np.ndarray'
    _theta_phi_mode_1_max_min: 'np.ndarray'
    _phi_phi_mode_1_max_min: 'np.ndarray'
    _mhr_mode_1_max_min: 'np.ndarray'
    _theta_der2_hr_mode_1_max_min: 'np.ndarray'
    _theta_der_hr_theta_mode_1_max_min: 'np.ndarray'
    _theta_der2_hr_theta_mode_1_max_min: 'np.ndarray'
    _theta_der_ht_theta_mode_1_max_min: 'np.ndarray'
    _theta_der_hp_theta_mode_1_max_min: 'np.ndarray'
    _hr_mode_2_max_min: 'np.ndarray'
    _ht_mode_2_max_min: 'np.ndarray'
    _hp_mode_2_max_min: 'np.ndarray'
    _theta_der_hr_mode_2_max_min: 'np.ndarray'
    _theta_der_ht_mode_2_max_min: 'np.ndarray'
    _theta_der_hp_mode_2_max_min: 'np.ndarray'
    _theta_phi_mode_2_max_min: 'np.ndarray'
    _phi_phi_mode_2_max_min: 'np.ndarray'
    _mhr_mode_2_max_min: 'np.ndarray'
    _theta_der2_hr_mode_2_max_min: 'np.ndarray'
    _theta_der_hr_theta_mode_2_max_min: 'np.ndarray'
    _theta_der2_hr_theta_mode_2_max_min: 'np.ndarray'
    _theta_der_ht_theta_mode_2_max_min: 'np.ndarray'
    _theta_der_hp_theta_mode_2_max_min: 'np.ndarray'
    _hr_mode_3_max_min: 'np.ndarray'
    _ht_mode_3_max_min: 'np.ndarray'
    _hp_mode_3_max_min: 'np.ndarray'
    _theta_der_hr_mode_3_max_min: 'np.ndarray'
    _theta_der_ht_mode_3_max_min: 'np.ndarray'
    _theta_der_hp_mode_3_max_min: 'np.ndarray'
    _theta_phi_mode_3_max_min: 'np.ndarray'
    _phi_phi_mode_3_max_min: 'np.ndarray'
    _mhr_mode_3_max_min: 'np.ndarray'
    _theta_der2_hr_mode_3_max_min: 'np.ndarray'
    _theta_der_hr_theta_mode_3_max_min: 'np.ndarray'
    _theta_der2_hr_theta_mode_3_max_min: 'np.ndarray'
    _theta_der_ht_theta_mode_3_max_min: 'np.ndarray'
    _theta_der_hp_theta_mode_3_max_min: 'np.ndarray'
    # normalized mode specific debug attributes
    _hr_mode_1_max_min_normed: 'np.ndarray'
    _ht_mode_1_max_min_normed: 'np.ndarray'
    _hp_mode_1_max_min_normed: 'np.ndarray'
    _theta_der_hr_mode_1_max_min_normed: 'np.ndarray'
    _theta_der_ht_mode_1_max_min_normed: 'np.ndarray'
    _theta_der_hp_mode_1_max_min_normed: 'np.ndarray'
    _theta_phi_mode_1_max_min_normed: 'np.ndarray'
    _phi_phi_mode_1_max_min_normed: 'np.ndarray'
    _mhr_mode_1_max_min_normed: 'np.ndarray'
    _theta_der2_hr_mode_1_max_min_normed: 'np.ndarray'
    _theta_der_hr_theta_mode_1_max_min_normed: 'np.ndarray'
    _theta_der2_hr_theta_mode_1_max_min_normed: 'np.ndarray'
    _theta_der_ht_theta_mode_1_max_min_normed: 'np.ndarray'
    _theta_der_hp_theta_mode_1_max_min_normed: 'np.ndarray'
    _hr_mode_2_max_min_normed: 'np.ndarray'
    _ht_mode_2_max_min_normed: 'np.ndarray'
    _hp_mode_2_max_min_normed: 'np.ndarray'
    _theta_der_hr_mode_2_max_min_normed: 'np.ndarray'
    _theta_der_ht_mode_2_max_min_normed: 'np.ndarray'
    _theta_der_hp_mode_2_max_min_normed: 'np.ndarray'
    _theta_phi_mode_2_max_min_normed: 'np.ndarray'
    _phi_phi_mode_2_max_min_normed: 'np.ndarray'
    _mhr_mode_2_max_min_normed: 'np.ndarray'
    _theta_der2_hr_mode_2_max_min_normed: 'np.ndarray'
    _theta_der_hr_theta_mode_2_max_min_normed: 'np.ndarray'
    _theta_der2_hr_theta_mode_2_max_min_normed: 'np.ndarray'
    _theta_der_ht_theta_mode_2_max_min_normed: 'np.ndarray'
    _theta_der_hp_theta_mode_2_max_min_normed: 'np.ndarray'
    _hr_mode_3_max_min_normed: 'np.ndarray'
    _ht_mode_3_max_min_normed: 'np.ndarray'
    _hp_mode_3_max_min_normed: 'np.ndarray'
    _theta_der_hr_mode_3_max_min_normed: 'np.ndarray'
    _theta_der_ht_mode_3_max_min_normed: 'np.ndarray'
    _theta_der_hp_mode_3_max_min_normed: 'np.ndarray'
    _theta_phi_mode_3_max_min_normed: 'np.ndarray'
    _phi_phi_mode_3_max_min_normed: 'np.ndarray'
    _mhr_mode_3_max_min_normed: 'np.ndarray'
    _theta_der2_hr_mode_3_max_min_normed: 'np.ndarray'
    _theta_der_hr_theta_mode_3_max_min_normed: 'np.ndarray'
    _theta_der2_hr_theta_mode_3_max_min_normed: 'np.ndarray'
    _theta_der_ht_theta_mode_3_max_min_normed: 'np.ndarray'
    _theta_der_hp_theta_mode_3_max_min_normed: 'np.ndarray'
    _p0_profile: 'np.ndarray'
    _rho0_profile: 'np.ndarray'
    _energy_mode_1: float
    _gamma_1_adDer_profile: 'np.ndarray'

    # class entity initialization
    def __init__(
        self,
        first_dict: 'dict[str, Any]',
        second_dict: 'dict[str, Any]',
        third_dict: 'dict[str, Any]',
        additional_profile_information: 'dict[str, Any]',
        freq_handler: FH,
        l_list: list[int],
        m_list: list[int],
        kwargs_diff_div: 'dict[str, Any] | None' = None,
        kwargs_diff_terms: 'dict[str, Any] | None' = None,
        diff_terms_method: str = 'gradient',
        use_complex: bool = False,
        polytropic: bool = False,
        analytic_polytrope: bool = False,
        polytrope_data_dict: 'dict[str, Any] | None' = None,
        npoints: int = 200,
        self_coupling: bool = False,
        use_brunt_def: bool = True,
        nr_omp_threads: int = 4,
        use_symbolic_derivatives: bool = True,
        use_parallel: bool = False,
        numerical_integration_method: str = 'trapz',
        conj_cc: bool | tuple[bool, bool, bool] | None = None,
        store_debug: bool = False,
        use_cheby_integration: bool = False,
        cheby_order_multiplier: int = 4,
        cheby_blow_up_protection: bool = False,
        cheby_blow_up_factor: float = 1.0e3,
    ) -> None:
        # store complex-valuedness attribute
        self._use_complex = use_complex
        # store the attribute that determines whether debug properties are stored
        self._store_debug = store_debug
        # store which of the modes is complex conjugated in the expression for the coupling coefficient
        self._conj_cc = conj_cc if conj_cc else (True, False, False)
        # store the frequency handler object
        self._freq_handler = freq_handler
        # store attributes related to using OpenMP for (parallelized) numerical integrations
        self._omp_threads = nr_omp_threads
        self._use_parallel = use_parallel
        # store the numerical integration method kwargs
        self._numerical_integration_method = numerical_integration_method
        self._use_cheby = use_cheby_integration
        self._cheby_order_mul = cheby_order_multiplier
        self._cheby_blow_up = cheby_blow_up_protection
        self._cheby_blow_up_fac = cheby_blow_up_factor
        # store symbolic derivative attribute
        self._use_symbolic_derivative = use_symbolic_derivatives
        # store the radial integration method
        self._radint = (
            self._compute_radial_integral_complex
            if use_complex
            else self._compute_radial_integral_real
        )
        # initialize the superclass
        super().__init__(
            nr_modes=3,
            kwargs_diff_div=kwargs_diff_div,
            kwargs_diff_terms=kwargs_diff_terms,
            diff_terms_method=diff_terms_method,
            polytropic=polytropic,
            store_debug=store_debug,
            analytic_polytrope=analytic_polytrope,
        )
        # store the common attributes for all the modes
        super()._extract_common_info_dictionary(my_dict=first_dict)
        # store the mode-dependent information stored in the information dictionaries as attributes
        dict_list: 'list[dict[str, Any]]' = [
            first_dict,
            second_dict,
            third_dict,
        ]
        for _i, _dict in enumerate(dict_list):
            super()._extract_mode_dependent_info_dictionary(
                my_dict=_dict, mode_number=_i + 1
            )
        # check if interpolations are necessary for the GYRE mode profiles
        super()._interpolation_modes(dict_list)
        # read and store the additional profile information
        super()._extract_additional_profile_info(
            my_internal_structure_dict=additional_profile_information
        )
        # make a distinction between actions necessary for polytropic models and MESA models
        if polytropic:
            # -- POLYTROPIC FILES
            # initialize the normalization factors
            super()._set_normalization_factors(
                P=self._p0_profile, rho=self._rho0_profile
            )
            # get additional attributes and derivatives
            if self._analytic_polytrope:
                # - from analytic expressions in the polytropes
                super()._get_profiles_polytrope_analytic(
                    my_dict=additional_profile_information
                )
            else:
                # - similar to MESA+GYRE files
                super()._get_profiles_polytrope_MESA_GYRE(
                    use_brunt_def=use_brunt_def
                )
            # get structure coefficients
            super()._store_polytropic_structure_coefficients()
        else:
            # -- MESA + GYRE FILES
            # initialize the normalization factors
            super()._set_normalization_factors(P=self._P, rho=self._rho)
            # store the gravitational constant in a uniform attribute
            self._G = first_dict['G']
            # get normed attributes and derivatives of MESA profiles
            super()._get_profiles_MESA_GYRE(use_brunt_def=use_brunt_def)
        # ensure that the center and surface point are cut out to avoid possible infinite values
        super()._mode_structure_cuts()
        # compute the spin factors TODO: check whether this is needed?!
        # super()._get_spin_factors()
        # store helper lists
        self._l_list = l_list
        self._m_list = m_list
        self._nu_list = list(self._freq_handler.spin_factors)
        # compute and store the radial vectors
        super()._compute_radial_vectors()
        # compute and store the radial gradients
        super()._compute_radial_gradients(diff_method=diff_terms_method)
        # compute the radial part of the divergence factors
        super()._compute_divergence_radial_part()
        # -- compute and store the lists to be used in the computation of symmetric S terms: additional layer of safety when storing the GYRE eigenvalues of the LTE's
        try:
            self._lambda_list = [
                getattr(self, f'_lambda_mode_{_g}') for _g in range(1, 4)
            ]  # GYRE eigenvalue of LTE's
        except AttributeError:
            # initialize the object
            _gl_obj = GyreLambdas(base_gyre_dir='')
            # use the object to retrieve the GYRE eigenvalues of the LTE's
            self._lambda_list = _gl_obj.get_lambda(
                self._l_list, self._m_list, self._nu_list
            )
        # store the number of points to be used in the computation of Hough functions
        self._npts = npoints
        # store whether you are computing self-coupling terms
        self._self_coupling = self_coupling
        # compute and store the Hough functions and their derivatives for the different modes
        self._store_houghs()
        # perform selected normalization of the Hough functions
        self._normalize_houghs()
        # compute the mode energies + b_a normalization factors
        self._compute_b_a_s_mode_energies()
        # normalize the radial eigenfunctions using a specific convention
        _enum_norm_object = (
            EnumRadialPoly if self._use_polytropic else EnumRadial
        )
        _norm_object: GYRENorm = GYRENorm(
            my_obj=self,
            coupler_enumeration=_enum_norm_object,
            norm_convention='lee2012',
            nr_modes=3,
            iterative_process=False,
            max_iterations=10,
        )
        _norm_object()  # call and perform normalization
        # store the normalizing factors
        self._normalizing_factors_modes = _norm_object.normalizing_factors
        # -- initialize the attribute that will hold the value of the coupling coefficient
        self._coupling_coefficient = None
        # -- initialize the attribute that will hold the coupling coefficient as a function of radius
        self._coupling_coefficient_integrated = None
        # fill up the debug attributes if necessary
        super()._fill_specific_mode_debug_attributes()
        self._fill_hough_mode_debug_attributes()

    # GETTER/SETTER METHODS
    @property
    def coupling_coefficient(self) -> None | float | complex:
        """Retrieves the (not-normalized) quadratic coupling coefficient.

        Notes
        -----
        The unit of the coupling coefficient is s-2.

        Returns
        -------
        float | complex | None
            The quadratic coupling coefficient.
        """
        return self._coupling_coefficient

    @property
    def coupling_coefficient_profile(self) -> 'np.ndarray | None':
        """Retrieves the profile of the coupling coefficient.

        Returns
        -------
        np.ndarray | None
            The profile of the coupling coefficient in function of the radius.
        """
        return self._coupling_coefficient_integrated

    @property
    def normed_coupling_coefficient(self) -> None | float | complex:
        """Retrieves the normalized quadratic coupling coefficient: dimensionless.

        Returns
        -------
        float | complex | None
            The quadratic coupling coefficient normalized by the mode energy.
        """
        if self._coupling_coefficient is None:
            return None
        else:
            return self._coupling_coefficient / self._energy_mode_1

    @property
    def normed_coupling_coefficient_profile(self) -> 'np.ndarray | None':
        """Retrieves the profile of the normalized quadratic coupling coefficient.

        Returns
        -------
        np.ndarray | None
            The profile of the normalized coupling coefficient.
        """
        if self._coupling_coefficient_integrated is None:
            return None
        else:
            return self._coupling_coefficient_integrated / self._energy_mode_1

    @property
    def normed_contributions_coupling_coefficient_profile(
        self
    ) -> 'tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None':
        """Retrieves the profile of the four contributing factors to the normalized quadratic coupling coefficient.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None
            The profile of the normalized coupling coefficient.
        """
        if self._coupling_coefficient_integrated is None:
            return None
        else:
            return (
                self._cc_term_1_integrated / self._energy_mode_1,
                self._cc_term_2_integrated / self._energy_mode_1,
                self._cc_term_3_integrated / self._energy_mode_1,
                self._cc_term_4_integrated / self._energy_mode_1,
            )

    @property
    def eta_lee_coupling_coefficient(self) -> None | float:
        """Retrieves the eta coupling coefficient defined in Lee (2012).

        Returns
        -------
        float or None
            The eta coupling coefficient, as defined in Lee (2012).
        """
        if self._coupling_coefficient is None:
            return None
        else:
            if TYPE_CHECKING:
                assert isinstance(self.normed_coupling_coefficient, float)
            return 2.0 * self.normed_coupling_coefficient

    @property
    def normalization_factors(self) -> np.ndarray:
        """Retrieve the normalization factors used to normalize the GYRE eigenfunctions of the modes in the mode triad.

        Returns
        -------
        np.ndarray
            The normalization factors for the modes in the mode triad.
        """
        return np.array(self._normalizing_factors_modes).reshape(
            -1,
        )

    @property
    def x(self) -> np.ndarray:
        """Retrieve the dimensionless radius profile.

        Returns
        -------
        np.ndarray
            The dimensionless radius profile.
        """
        return self._x

    @property
    def coupling_terms_quadratic(self) -> np.ndarray:
        """Retrieve the individual coupling terms for the g-mode quadratic coupling coefficient of a rotating star.

        Notes
        -----
        See Lee (2012) and Van Beeck et al. (forthcoming) for additional information on the coupling terms.

        Returns
        -------
        coupling_term_arr : np.ndarray
            Contains the 4 terms used to compute the coupling coefficient (4 terms summed).
        """
        # create and return the coupling term array
        return np.array([getattr(self, f'_cc_term_{x}') for x in range(1, 5)])

    # internal utility method that stores the Hough functions for the different modes
    def _store_houghs(self) -> None:
        """Internal utility method that stores the necessary Hough function for the mode coupling coefficient computations."""
        if TYPE_CHECKING:
            assert isinstance(self._l_list, list) and len(self._l_list) == 3
            assert isinstance(self._m_list, list) and len(self._m_list) == 3
            assert isinstance(self._nu_list, list) and len(self._nu_list) == 3
            assert (
                isinstance(self._lambda_list, list)
                and len(self._lambda_list) == 3
            )
        # loop over the necessary information of the 3 modes, and compute the relevant Hough functions + derivatives for all
        for _i, _l, _m, _nu, _lmb in zip(
            range(1, 4),
            self._l_list,
            self._m_list,
            self._nu_list,
            self._lambda_list,
        ):
            # obtain the relevant information on the Hough functions
            (
                _hough_dict,
                _mu_values,
                _theta_values,
            ) = HFHandler.get_relevant_hough_output(
                nu=_nu,
                l=_l,
                m=_m,
                npts=self._npts,
                lmbd=_lmb,
                use_complex=self._use_complex,
                lmbd_adjust=True,
                definition='lee_saio_equivalent',
            )
            # unpack the dictionary and store the items
            for _key, _val in _hough_dict.items():
                setattr(self, f'_{_key}_mode_{_i}', _val)
                # set debug attributes, if necessary
                if self._store_debug:
                    self._add_debug_attribute(f'_{_key}_mode_{_i}')
        # store the mu and theta values
        if TYPE_CHECKING:
            # Fix for unbound local error (these variables are always bound!)
            _mu_values: Any = None
            _theta_values: Any = None
        self._mu_values = _mu_values
        self._theta_values = _theta_values
        # add a debug attribute for the mu and theta values (should not be an issue), if necessary
        if self._store_debug:
            self._add_debug_attribute('_mu_values')
            self._add_debug_attribute('_theta_values')

    # internal utility method that performs the normalization of the Hough functions
    def _normalize_houghs(self, normalization_method: str = 'lee2012') -> None:
        """Internal utility method that normalizes the Hough functions (+ corresponding derivatives).

        Parameters
        ----------
        normalization_method : str, optional
            Selects the appropriate normalization method, options: ['lee2012', 'prat2019'], by default 'lee2012'.
        """
        # store the names of the attributes to be normalized
        _attrs_to_norm = HFHandler.get_functional_names()
        # loop over the radial Hough functions and retrieve the appropriate normalization factors
        for _i in range(1, 4):
            # retrieve the normalization factor
            _norm_fac = HNorm.get_norm_factor(
                norm_procedure=normalization_method,
                mu=self._mu_values,
                hr=getattr(self, f'_hr_mode_{_i}'),
                nr_omp_threads=self._omp_threads,
                use_parallel=self._use_parallel,
            )
            # perform the normalizations
            for _attr in _attrs_to_norm:
                # get the name of the attribute to be stored
                _name = f'_{_attr}_mode_{_i}'
                # store the normalized attribute
                setattr(self, _name, getattr(self, _name) * _norm_fac)
                # store (normalized) debug attributes, if necessary
                if self._store_debug:
                    self._add_debug_attribute(
                        f'_{_attr}_mode_{_i}', normed=True
                    )

    # internal utility method that performs the integration of the radial terms
    def _rad_int(
        self,
        radial_terms: list[str] | tuple[str, ...],
        multiplying_term: np.ndarray,
        indexer: slice = np.s_[:],
    ) -> float | np.ndarray:
        """Internal utility method used to perform radial integration.

        Parameters
        ----------
        radial_terms : list[str]
            Contains the strings that identify different radial terms.
        multiplying_term : np.ndarray[float]
            The term with which the integrand needs to be multiplied in order to compute the integral.
        indexer : np.s_, optional
            Numpy slicing object used to retrieve a specific term of the radial coupling coefficient profile; by default np.s_[:].

        Returns
        -------
        float
            The result of the radial integration.
        """
        # get the term array
        first_attr = getattr(self, f'{radial_terms[0]}_mode_1')
        _term_arr = np.empty(
            (first_attr.shape[0], self._nr_modes), dtype=np.complex128
        )
        _term_arr[:, 0] = first_attr
        for _i, _rt in enumerate(radial_terms[1:]):
            _term_arr[:, _i + 1] = getattr(self, f'{_rt}_mode_{_i + 2}')
        # compute the integrand, multiplied with the additional radial profile term and the spherical volume term
        _integrand = (
            np.prod(_term_arr, axis=1) * multiplying_term * self._x ** (2.0)
        )
        # integrate over the integrand and return the result
        return self._radint(_integrand, self._x, indexer)

    # internal utility method that maps Hough descriptions to specific modes
    def _map_hough_to_mode(
        self, my_h_list: list[str], mapping_indices: list[int]
    ) -> list[str]:
        """Internal utility method used to map Hough function (or derivatives) descriptions to the input to compute the corresponding integrals/integrands.

        Parameters
        ----------
        my_h_list : list[str]
            The Hough function (or derivative) descriptors.
        mapping_indices : list[int]
            The indices used to map the specific modes to the Hough functions in the integrands/integrals.

        Returns
        -------
        list[str]
            List of mapping strings.
        """
        if len(my_h_list) == len(mapping_indices):
            return [
                f'_{_h}_mode_{_mi}'
                for _h, _mi in zip(my_h_list, mapping_indices)
            ]
        else:
            logger.error('Length of inputs is not equal. Now exiting.')
            sys.exit()

    # internal utility method that maps symmetric terms
    @staticmethod
    def _symmetry_mapping(
        operator_list: list[str]
    ) -> defaultdict[tuple[str, ...], list]:
        """Internal utility method that generates a mapping of the operators in the operator_list to its unique components, uncovering possible symmetries.

        Parameters
        ----------
        operator_list : list[str]
            Describes the operators.

        Returns
        -------
        operator_dict : defaultdict[tuple[str, ...], list]
            Maps unique components to their indices in the symmetrized list.
        """
        # generate the symmetric permutation list
        _sym_trans = list(permutations(operator_list))
        # use that list to create the mapping dictionary (maps unique components to their indices in the symmetrized list)
        operator_dict = defaultdict(list)
        for _i, _tupe in enumerate(_sym_trans):
            operator_dict[_tupe].append(_i)
        # return the mapping dictionary
        return operator_dict

    # internal utility (dispatched) method that computes the symmetric terms that need to be computed
    @multimethod
    def _overloaded_symmetry_determiner(  # type: ignore
        self,
        angular_terms: list[str],
        radial_terms: list[str],
        div_by_sin: bool,
        radial_multiplier,
        ang_factor=[1.0],
        rad_factor=[1.0],
        rad_indexer=np.s_[:],
    ) -> float | complex:
        """Internal utility dispatched method used to compute the result of the S operator (accounting for S operator symmetry + additional angular terms) for the non-self-coupling case.

        Parameters
        ----------
        angular_terms : list[str]
            Contains the strings that identify different angular terms.
        radial_terms : list[str]
            Contains the strings that identify different radial terms.
        div_by_sin : bool
            If True, divide the integrand by sin(theta) before integrating. If False, do not do so.
        radial_multiplier : np.ndarray[float]
            The additional radial multiplication term (for the integrand).
        ang_factor : list[float], optional
            The list of multiplication terms for the angular integrals; by default [1.0].
        rad_factor : list[float], optional
            The list of multiplication terms for the radial integrals (outside the integral); by default [1.0].
        rad_indexer : np.s_, optional
            Numpy slicing object used to retrieve a specific term of the radial coupling coefficient profile; by default np.s_[:].

        Returns
        -------
        float or complex
            The result of the S operation and subsequent integration.
        """
        # initialize result arrays
        _rad_results, _ang_results = np.empty(6), np.empty(6)
        # get symmetrization dictionaries
        _rad_dict = self._symmetry_mapping(radial_terms)
        _ang_dict = self._symmetry_mapping(angular_terms)
        # fill up the result arrays
        # - angular
        for _key, _index in _ang_dict.items():
            _ang_results[_index] = ang_factor[
                0
            ] * HI.quadratic_angular_integral(
                my_obj=self,
                m_list_for_azimuthal_check=self._m_list,
                conj=self._conj_cc,
                descrips=self._map_hough_to_mode(list(_key), [1, 2, 3]),
                div_by_sin=div_by_sin,
                mul_with_mu=False,
                nr_omp_threads=self._omp_threads,
                use_parallel=self._use_parallel,
                num_integ_method=self._numerical_integration_method,
                use_cheby_integration=self._use_cheby,
                cheby_order_multiplier=self._cheby_order_mul,
                cheby_blow_up_protection=self._cheby_blow_up,
                cheby_blow_up_factor=self._cheby_blow_up_fac,
            )
        # - radial
        for _key, _index in _rad_dict.items():
            _rad_results[_index] = rad_factor[0] * self._rad_int(
                radial_terms=_key,
                multiplying_term=radial_multiplier,
                indexer=rad_indexer,
            )
        # now compute the end result of applying the S operator and return it
        return np.dot(_ang_results, _rad_results)

    @_overloaded_symmetry_determiner.register(list[str], list[list[str]], bool)  # type: ignore
    @_overloaded_symmetry_determiner.register(  # type: ignore
        list[str], list[tuple[str, ...]], bool
    )
    def _(
        self,
        angular_terms,
        radial_terms,
        div_by_sin,
        radial_multiplier,
        ang_factor=[1.0],
        rad_factor=[1.0],
        rad_indexer=np.s_[:],
    ) -> float | complex:
        """Internal utility dispatched method used to compute the result of the S operator (accounting for S operator symmetry + additional angular terms) for the non-self-coupling case.

        Parameters
        ----------
        angular_terms : list[str]
            Contains the strings that identify different angular terms.
        radial_terms : list[list[str]] | list[tuple[str, ...]]
            Contains the strings that identify different radial terms.
        radial_multiplier : np.ndarray[float]
            The additional radial multiplication term.
        div_by_sin : bool
            If True, divide the integrand by sin(theta) before integrating. If False, do not do so.
        ang_factor : list[float], optional
            The multiplication factor for the angular integration terms. Takes care of sign changes as well; by default [1.0].
        rad_factor : list[float], optional
            The list of multiplication terms for the radial integrals (outside the integral); by default [1.0].
        rad_indexer : np.s_, optional
            Numpy slicing object used to retrieve a specific term of the radial coupling coefficient profile; by default np.s_[:].

        Returns
        -------
        float or complex
            The result of the S operation and subsequent integration.
        """
        # initialize result arrays
        _rad_results, _ang_results = (
            np.empty((6, len(radial_terms))),
            np.empty(6),
        )
        # get symmetrization dictionary
        _ang_dict = self._symmetry_mapping(angular_terms)
        # fill up the result arrays
        # - angular
        for _key, _index in _ang_dict.items():
            _ang_results[_index] = ang_factor[
                0
            ] * HI.quadratic_angular_integral(
                my_obj=self,
                m_list_for_azimuthal_check=self._m_list,
                conj=self._conj_cc,
                descrips=self._map_hough_to_mode(list(_key), [1, 2, 3]),
                div_by_sin=div_by_sin,
                mul_with_mu=False,
                nr_omp_threads=self._omp_threads,
                use_parallel=self._use_parallel,
                num_integ_method=self._numerical_integration_method,
                use_cheby_integration=self._use_cheby,
                cheby_order_multiplier=self._cheby_order_mul,
                cheby_blow_up_protection=self._cheby_blow_up,
                cheby_blow_up_factor=self._cheby_blow_up_fac,
            )
        # - radial
        for _i, _rad_t in enumerate(radial_terms):
            # get the symmetrization dictionary
            _rad_dict = self._symmetry_mapping(_rad_t)
            # compute and store the results
            for _key, _index in _rad_dict.items():
                _rad_results[_index, _i] = rad_factor[_i] * self._rad_int(
                    radial_terms=_key,
                    multiplying_term=radial_multiplier,
                    indexer=rad_indexer,
                )
        # now compute the end result of applying the S operator and return it
        return (_rad_results * _ang_results[:, np.newaxis]).sum()

    @multimethod
    def _overloaded_symmetry_determiner(
        self,
        angular_terms: list[list[str]],
        radial_terms: list[str],
        div_by_sin: list[bool],
        radial_multiplier,
        ang_factor=[1.0],
        rad_factor=[1.0],
        rad_indexer=np.s_[:],
    ) -> float | complex:
        """Internal utility dispatched method used to compute the result of the S operator (accounting for S operator symmetry + additional angular terms) for the non-self-coupling case.

        Parameters
        ----------
        angular_terms : list[list[str]]
            Contains the strings that identify different angular terms.
        radial_terms : list[str]
            Contains the strings that identify different radial terms.
        radial_multiplier : np.ndarray[float]
            The additional radial multiplication term.
        div_by_sin : list[bool]
            Contains values that if True, divide the specific angular integrand (specified in a list of list[str] in the angular_terms) by sin(theta) before integrating (if False, no such division is done).
        ang_factor : list[float], optional
            The multiplication factor for the angular integration terms. Takes care of sign changes as well; by default [1.0].
        rad_factor : list[float], optional
            The list of multiplication terms for the radial integrals (outside the integral); by default [1.0].
        rad_indexer : np.s_, optional
            Numpy slicing object used to retrieve a specific term of the radial coupling coefficient profile; by default np.s_[:].

        Returns
        -------
        float or complex
            The result of the S operation and subsequent integration.
        """
        # initialize result arrays
        _rad_results, _ang_results = (
            np.empty(6),
            np.empty((6, len(angular_terms))),
        )
        # get symmetrization dictionary
        _rad_dict = self._symmetry_mapping(radial_terms)
        # fill up the result arrays
        # - radial
        for _key, _index in _rad_dict.items():
            _rad_results[_index] = rad_factor[0] * self._rad_int(
                radial_terms=_key,
                multiplying_term=radial_multiplier,
                indexer=rad_indexer,
            )
        # - angular
        for _i, _ang in enumerate(angular_terms):
            # get the symmetrization dictionary
            _ang_dict = self._symmetry_mapping(_ang)
            # compute the angular integration results
            for _key, _index in _ang_dict.items():
                _ang_results[_index, _i] = ang_factor[
                    _i
                ] * HI.quadratic_angular_integral(
                    my_obj=self,
                    m_list_for_azimuthal_check=self._m_list,
                    conj=self._conj_cc,
                    descrips=self._map_hough_to_mode(list(_key), [1, 2, 3]),
                    div_by_sin=div_by_sin[_i],
                    mul_with_mu=False,
                    nr_omp_threads=self._omp_threads,
                    use_parallel=self._use_parallel,
                    num_integ_method=self._numerical_integration_method,
                    use_cheby_integration=self._use_cheby,
                    cheby_order_multiplier=self._cheby_order_mul,
                    cheby_blow_up_protection=self._cheby_blow_up,
                    cheby_blow_up_factor=self._cheby_blow_up_fac,
                )
        # now compute the end result of applying the S operator and return it
        return (_rad_results[:, np.newaxis] * _ang_results).sum()

    @_overloaded_symmetry_determiner.register(  # type: ignore
        list[list[str]], list[list[str]], list[bool]
    )
    @_overloaded_symmetry_determiner.register(  # type: ignore
        list[list[str]], list[tuple[str, ...]], list[bool]
    )
    def _(
        self,
        angular_terms,
        radial_terms,
        div_by_sin,
        radial_multiplier,
        ang_factor=[1.0],
        rad_factor=[1.0],
        rad_indexer=np.s_[:],
    ) -> float | complex:
        """Internal utility dispatched method used to compute the result of the S operator (accounting for S operator symmetry + additional angular terms) for the non-self-coupling case.

        Parameters
        ----------
        angular_terms : list[list[str]]
            Contains the strings that identify different angular terms.
        radial_terms : list[list[str]] | list[tuple[str, ...]]
            Contains the strings that identify different radial terms.
        radial_multiplier : np.ndarray[float]
            The additional radial multiplication term.
        div_by_sin : list[bool]
            Contains values that if True, divide the specific angular integrand (specified in a list of list[str] in the angular_terms) by sin(theta) before integrating (if False, no such division is done).
        ang_factor : list[float], optional
            The multiplication factor for the angular integration terms. Takes care of sign changes as well; by default [1.0].
        rad_factor : list[float], optional
            The list of multiplication terms for the radial integrals (outside the integral); by default [1.0].
        rad_indexer : np.s_, optional
            Numpy slicing object used to retrieve a specific term of the radial coupling coefficient profile; by default np.s_[:].

        Returns
        -------
        float or complex
            The result of the S operation and subsequent integration.
        """
        # initialize result arrays
        _rad_results, _ang_results = (
            np.empty((6, len(radial_terms))),
            np.empty((6, len(angular_terms))),
        )
        # fill up the result arrays
        # - angular
        for _j, _ang in enumerate(angular_terms):
            # get the symmetrization dictionary
            _ang_dict = self._symmetry_mapping(_ang)
            # compute and store the results
            for _key, _index in _ang_dict.items():
                _ang_results[_index, _j] = ang_factor[
                    _j
                ] * HI.quadratic_angular_integral(
                    my_obj=self,
                    m_list_for_azimuthal_check=self._m_list,
                    conj=self._conj_cc,
                    descrips=self._map_hough_to_mode(list(_key), [1, 2, 3]),
                    div_by_sin=div_by_sin[_j],
                    mul_with_mu=False,
                    nr_omp_threads=self._omp_threads,
                    use_parallel=self._use_parallel,
                    num_integ_method=self._numerical_integration_method,
                    use_cheby_integration=self._use_cheby,
                    cheby_order_multiplier=self._cheby_order_mul,
                    cheby_blow_up_protection=self._cheby_blow_up,
                    cheby_blow_up_factor=self._cheby_blow_up_fac,
                )
        # - radial
        for _i, _rad_t in enumerate(radial_terms):
            # get the symmetrization dictionary
            _rad_dict = self._symmetry_mapping(_rad_t)
            # compute and store the results
            for _key, _index in _rad_dict.items():
                _rad_results[_index, _i] = rad_factor[_i] * self._rad_int(
                    radial_terms=_key,
                    multiplying_term=radial_multiplier,
                    indexer=rad_indexer,
                )
        # now compute the end result of applying the S operator and return it
        return np.einsum('ji,jk->ik', _rad_results, _ang_results).sum()

    # method that implements the symmetrization terms
    def _symmetrization_s_term_integral(
        self,
        radial_terms: list[str] | list[tuple[str, ...]],
        radial_multiplier: np.ndarray,
        angular_terms: list[str] | list[list[str]],
        div_by_sin: bool | list[bool] = False,
        ang_factor: list[float] = [1.0],
        rad_factor: list[float] = [1.0],
        rad_indexer: slice = np.s_[:],
    ) -> float | complex | np.ndarray:
        """Internal method that implements the integral of the symmetric S operator defined in equations B22 and B24 of Lee(2012), which will be used for computation of integrands for quadratic coupling coefficients.

        Notes
        -----
        In addition to the angular integral taken in expression B24 of Lee(2012), we also immediately perform radial integration.

        Parameters
        ----------
        radial_terms : list[str] or list[list, str]
            Contains the strings that identify different radial terms.
        radial_multiplier : np.ndarray[float]
            The additional radial multiplication term.
        angular_terms : list[str] or list[list, str]
            Contains the strings that identify different angular terms.
        div_by_sin : bool | list[bool], optional
            If True, divide the angular integrand by sin(theta) before integrating. Default: False.
        ang_factor : list[float], optional
            The multiplication factor for the angular integration terms. Takes care of sign changes as well; by default [1.0].
        rad_factor : list[float], optional
            The list of multiplication terms for the radial integrals (outside the integral); by default [1.0].
        rad_indexer : np.s_, optional
            Numpy slicing object used to retrieve a specific term of the radial coupling coefficient profile; by default np.s_[:].

        Returns
        -------
        float or complex
            The result of the S operation and subsequent integration.
        """
        # - self-coupling case
        if self._self_coupling:
            # only one volume integral needs to be computed!
            # -- compute the angular integral part(s)
            if isinstance(angular_terms[0], list) and isinstance(
                div_by_sin, list
            ):
                _ang_integration_result = 0.0
                # compute the angular integration result
                for _ang_term, _div_by_sin_term in zip(
                    angular_terms, div_by_sin
                ):
                    if TYPE_CHECKING:
                        assert isinstance(_ang_term, list)
                    _ang_integration_result += HI.quadratic_angular_integral(
                        my_obj=self,
                        m_list_for_azimuthal_check=self._m_list,
                        conj=self._conj_cc,
                        descrips=self._map_hough_to_mode(_ang_term, [1, 2, 3]),
                        div_by_sin=_div_by_sin_term,
                        mul_with_mu=False,
                        nr_omp_threads=self._omp_threads,
                        use_parallel=self._use_parallel,
                        num_integ_method=self._numerical_integration_method,
                        use_cheby_integration=self._use_cheby,
                        cheby_order_multiplier=self._cheby_order_mul,
                        cheby_blow_up_protection=self._cheby_blow_up,
                        cheby_blow_up_factor=self._cheby_blow_up_fac,
                    )
            else:
                if TYPE_CHECKING:
                    assert isinstance(div_by_sin, bool)
                _ang_integration_result = HI.quadratic_angular_integral(
                    my_obj=self,
                    m_list_for_azimuthal_check=self._m_list,
                    conj=self._conj_cc,
                    descrips=self._map_hough_to_mode(angular_terms, [1, 2, 3]),  # type: ignore (no clue how to convince pyright)
                    div_by_sin=div_by_sin,
                    mul_with_mu=False,
                    nr_omp_threads=self._omp_threads,
                    use_parallel=self._use_parallel,
                    num_integ_method=self._numerical_integration_method,
                    use_cheby_integration=self._use_cheby,
                    cheby_order_multiplier=self._cheby_order_mul,
                    cheby_blow_up_protection=self._cheby_blow_up,
                    cheby_blow_up_factor=self._cheby_blow_up_fac,
                )
            # -- return a zero integral result if the angular integral is zero due to e.g. selection rules
            if _ang_integration_result == 0:
                # return the integration result: zero
                return 0.0
            else:
                # -- compute the radial integral part
                if isinstance(radial_terms[0], list):
                    _rad_integration_result = 0.0
                    # compute the radial integration result
                    for _rad_term in radial_terms:
                        if TYPE_CHECKING:
                            assert isinstance(_rad_term, list)
                        _rad_integration_result += self._rad_int(
                            radial_terms=_rad_term,
                            indexer=rad_indexer,
                            multiplying_term=radial_multiplier,
                        )
                else:
                    _rad_integration_result = self._rad_int(
                        radial_terms=radial_terms,  # type: ignore (I don't know how to convince pyright this is of the right type!)
                        indexer=rad_indexer,
                        multiplying_term=radial_multiplier,
                    )
                # return the integration result: non-zero
                return _rad_integration_result * _ang_integration_result * 6.0
        # - other symmetry cases
        else:
            return self._overloaded_symmetry_determiner(
                angular_terms,
                radial_terms,
                div_by_sin,
                radial_multiplier=radial_multiplier,
                ang_factor=ang_factor,
                rad_factor=rad_factor,
                rad_indexer=rad_indexer,
            )

    # method that computes the classical norm
    def _compute_traditional_norms_modes(self, mode_nr: int) -> None:
        """Internal method that computes the traditional norm of the eigenfunctions.

        Parameters
        ----------
        mode_nr : int
            The integer that retrieves the correct information from the information lists.
        """
        # -- compute the 'traditional norm' of the mode (= norm of the mode when there is no rotation)
        # - initialize variable
        trad_norm = 0.0
        # - initialize lists containing instructions
        _h_list_trad_norm = [['hr', 'hr'], ['ht', 'ht'], ['hp', 'hp']]
        _r_list_trad_norm = [
            f'_x_z_1_mode_{mode_nr}',
            f'_x_z_2_over_c_1_omega_mode_{mode_nr}',
            f'_x_z_2_over_c_1_omega_mode_{mode_nr}',
        ]
        # - compute common prefactor
        _prefac = self._x ** (2.0) * self._rho_normed
        # compute the necessary integrals
        for _h_list, _r_comp in zip(_h_list_trad_norm, _r_list_trad_norm):
            # compute the angular term --> NO CONJUGATIONS
            _trad_norm_angular = HI.quadratic_angular_integral(
                my_obj=self,
                conj=False,
                div_by_sin=False,
                descrips=self._map_hough_to_mode(_h_list, [mode_nr, mode_nr]),
                nr_omp_threads=self._omp_threads,
                mul_with_mu=False,
                use_parallel=self._use_parallel,
                m_list_for_azimuthal_check=None,
                num_integ_method=self._numerical_integration_method,
                use_cheby_integration=self._use_cheby,
                cheby_order_multiplier=self._cheby_order_mul,
                cheby_blow_up_protection=self._cheby_blow_up,
                cheby_blow_up_factor=self._cheby_blow_up_fac,
            )
            # compute the radial term integrand
            _trad_norm_radial_integrand = _prefac * getattr(self, _r_comp) ** (
                2.0
            )
            # compute the radial term integral
            _trad_norm_radial = self._radint(
                _trad_norm_radial_integrand, self._x
            )
            # compute the result of the specific term in the 'traditional norm' and add it to the total
            trad_norm += _trad_norm_angular * _trad_norm_radial
        # multiply with pre-factor (due to Phi coordinate integration) to get the 'traditional norm' and store in the object
        setattr(self, f'_classic_norm_mode_{mode_nr}', trad_norm * 2.0 * np.pi)

    # generic method that computes a radial integral
    def _compute_radial_integral(
        self,
        integrand: np.ndarray,
        integrating_quantity: np.ndarray,
        indexer: slice = np.s_[:],
    ) -> np.ndarray:
        """Internal method used to compute a radial integral, in which the integrand and integrating quantity are specified as input.

        Parameters
        ----------
        integrand : np.ndarray
            Real-valued integrand array.
        integrating_quantity : np.ndarray
            Real-valued integrating quantity (radius e.g.).
        indexer : np.s_, optional
            Numpy slicing object used to retrieve a specific term of the radial coupling coefficient profile; by default np.s_[:].

        Returns
        -------
        result_array : np.ndarray
            The integrated quantity, real-valued.
        """
        # perform the numerical integration and return the result
        return cni.integrate(
            self._numerical_integration_method,
            False,
            integrand[indexer],
            integrating_quantity[indexer],
            self._omp_threads,
            self._use_parallel,
        )

    # generic method that computes the complex-valued radial integral
    def _compute_radial_integral_complex(
        self: 'Self',
        integrand: np.ndarray,
        integrating_quantity: np.ndarray,
        indexer: slice = np.s_[:],
    ) -> float | np.ndarray:
        """Internal method used to compute a radial integral, in which the complex-valued integrand and integrating quantity are specified as input.

        Parameters
        ----------
        integrand : np.ndarray
            Complex-valued integrand array.
        integrating_quantity : np.ndarray
            Real-valued integrating quantity (radius e.g.).
        indexer : np.s_, optional
            Numpy slicing object used to retrieve a specific term of the radial coupling coefficient profile; by default np.s_[:].

        Returns
        -------
        result_array : np.ndarray
            The integrated quantity, complex-valued.
        """
        # compute the real and imaginary parts of the integral
        _real = self._compute_radial_integral(
            np.ascontiguousarray(integrand.real), integrating_quantity, indexer
        )
        _imag = self._compute_radial_integral(
            np.ascontiguousarray(integrand.imag), integrating_quantity, indexer
        )
        # retrieve and return the result array
        result_array = np.empty((_imag.shape[0],), dtype=np.complex128)
        result_array.real, result_array.imag = _real, _imag
        # return the complex-valued result
        return result_array

    # generic method that computes the real-valued radial integral
    def _compute_radial_integral_real(
        self,
        integrand: np.ndarray,
        integrating_quantity: np.ndarray,
        indexer: slice = np.s_[:],
    ) -> float | np.ndarray:
        """Generic internal method used to compute a radial integral, in which the (real-valued) integrand and integrating quantity are specified as input.

        Parameters
        ----------
        integrand : np.ndarray
            Real-valued integrand array.
        integrating_quantity : np.ndarray
            Real-valued integrating quantity (radius e.g.).
        indexer : np.s_, optional
            Numpy slicing object used to retrieve a specific term of the radial coupling coefficient profile; by default np.s_[:].

        Returns
        -------
        result_array : np.ndarray
            The integrated quantity, real-valued.
        """
        # return the real-valued result
        return self._compute_radial_integral(
            np.ascontiguousarray(integrand.real), integrating_quantity, indexer
        )

    # method that computes the mode energies
    def _compute_b_a_s_mode_energies(self) -> None:
        """Internal method that computes the normalization factors b_a and the mode energies epsilon for each of the modes."""
        # loop over the different modes + store the necessary information
        for _i in range(1, 4):
            self.compute_b_a_mode_energy(mode_nr=_i)

    # method that computes the b_a normalization factor (assuming a time dependence of e^(-i * omega * t)), as well as the mode (oscillation) energy
    def compute_b_a_mode_energy(self, mode_nr: int) -> None:
        """Method that computes the normalization factor b_a (see Schenk et al. (2001) and Lee(2012)) used for the normalization of the mode eigenfunctions and computation of the mode oscillation energies, assuming a time dependence of e^(-i*omega*t). (similar to Schenk et al. (2002) and Prat et al. (2019))

        Parameters
        ----------
        mode_nr : int
            The integer that retrieves the correct information from the information lists.
        """
        # compute the radial pre-factor in order to go from dimensionless to dimensioned energy
        _dimension_radial_prefactor = self._norm_rho * self._R_star ** (5.0)
        # -- compute the 'traditional norm' of the mode (= norm of the mode when there is no rotation)
        self._compute_traditional_norms_modes(mode_nr=mode_nr)
        # -- compute the 'additional (Coriolis) term' in the normalization due to rotation
        # - compute the angular integral (NO CONJUGATION)
        _coriolis_norm_angular = HI.quadratic_angular_integral(
            my_obj=self,
            conj=None,
            div_by_sin=False,
            mul_with_mu=True,
            m_list_for_azimuthal_check=None,
            descrips=self._map_hough_to_mode(['ht', 'hp'], [mode_nr, mode_nr]),
            nr_omp_threads=self._omp_threads,
            use_parallel=self._use_parallel,
            num_integ_method=self._numerical_integration_method,
            use_cheby_integration=self._use_cheby,
            cheby_order_multiplier=self._cheby_order_mul,
            cheby_blow_up_protection=self._cheby_blow_up,
            cheby_blow_up_factor=self._cheby_blow_up_fac,
        )
        # compute the radial integral
        _coriolis_norm_radial_integrand = (
            self._rho_normed
            * (self._x**2.0)
            * getattr(self, f'_x_z_2_over_c_1_omega_mode_{mode_nr}') ** 2.0
        )
        _coriolis_norm_radial = self._radint(
            _coriolis_norm_radial_integrand, self._x
        )
        # compute the total Coriolis term contribution
        _coriolis_norm = (
            _coriolis_norm_radial
            * _coriolis_norm_angular
            * 8.0
            * np.pi
            * self._freq_handler.surf_rot_angular_freq
        )
        # get the corotating-frame frequency for the modes
        my_corot = self._freq_handler.corot_mode_omegas[mode_nr - 1]
        # -- compute and store the complete normalization factor b_a assuming a time dependence of e^(-i*omega*t)
        b_a = (
            _coriolis_norm
            + (2.0 * my_corot * getattr(self, f'_classic_norm_mode_{mode_nr}'))
        ) * _dimension_radial_prefactor  # create local variable
        setattr(self, f'_b_a_mode_{mode_nr}', b_a)  # store b_a
        # -- compute and store the oscillation mode energy
        setattr(self, f'_energy_mode_{mode_nr}', my_corot * b_a)

    # METHODS COMPUTING THE RADIAL TERMS OF THE COUPLING COEFFICIENTS OF THE THREE-WAVE COUPLING
    # - Kappa(1)_ABC -> similar to Eq.(B2) in Lee(2012)
    def _kappa_1_abc(self, indexer: slice) -> float:
        """Internal utility method used to compute term Kappa^1_ABC (similar to Eq.(B2) in Lee, 2012)

        Parameters
        ----------
        indexer : np.s_
            Numpy slicing object used to retrieve a specific term of the coupling coefficient profile.

        Returns
        -------
        float
            The (dimensionalized) integration result.
        """
        # compute the radial multiplier term
        radial_multiplier = self._P_normed * (self._Gamma_1 - 1.0)
        # compute the dimensionalizing term
        _dim_term = self._norm_P * self._R_star ** (3.0)
        # compute the symmetric terms
        # - see Lee (2012), Eq. (B17)
        _sym1 = (
            0.5
            * self._symmetrization_s_term_integral(  # first term of Lee(2012)
                radial_terms=[
                    '_rad_der_x_z_1',
                    '_rad_der_x_z_1',
                    '_rad_diverg',
                ],
                angular_terms=['hr', 'hr', 'hr'],
                radial_multiplier=radial_multiplier,
                rad_indexer=indexer,
            )
        )
        _sym2 = (
            0.5
            * self._symmetrization_s_term_integral(  # second term of Lee(2012)
                radial_terms=[
                    '_z_2_over_c_1_omega',
                    '_z_2_over_c_1_omega',
                    '_rad_diverg',
                ],
                angular_terms=[
                    ['theta_der_ht_theta', 'theta_der_ht_theta', 'hr'],
                    ['phi_phi', 'phi_phi', 'hr'],
                    ['theta_phi', 'theta_der_hp_theta', 'hr'],
                ],
                radial_multiplier=radial_multiplier,
                rad_indexer=indexer,
                div_by_sin=[False, False, True],
                ang_factor=[1.0, 1.0, -2.0],
            )
        )
        _sym3 = self._symmetrization_s_term_integral(  # third term of Lee(2012)
            radial_terms=['_z_1', '_z_1', '_rad_diverg'],
            rad_indexer=indexer,
            angular_terms=['hr', 'hr', 'hr'],
            radial_multiplier=radial_multiplier,
        )
        _sym4 = (
            self._symmetrization_s_term_integral(  # fourth term of Lee(2012)
                radial_terms=['_z_2_over_c_1_omega', '_z_1', '_rad_diverg'],
                angular_terms=[
                    ['theta_der_ht_theta', 'hr', 'hr'],
                    ['phi_phi', 'hr', 'hr'],
                ],
                radial_multiplier=radial_multiplier,
                div_by_sin=[False, False],
                ang_factor=[1.0, 1.0],
                rad_indexer=indexer,
            )
        )
        _sym5 = self._symmetrization_s_term_integral(  # fifth term of Lee(2012)
            radial_terms=['_z_1', '_rad_der_x_z_2_c_1_omega', '_rad_diverg'],
            angular_terms=[
                ['theta_der_hr_theta', 'ht', 'hr'],
                ['mhr', 'hp', 'hr'],
            ],
            radial_multiplier=radial_multiplier,
            div_by_sin=[False, True],
            ang_factor=[1.0, -1.0],
            rad_indexer=indexer,
        )
        _sym6 = self._symmetrization_s_term_integral(  # sixth term of Lee(2012)
            radial_terms=[
                '_rad_diverg',
                '_z_2_over_c_1_omega',
                '_rad_der_x_z_2_c_1_omega',
            ],
            angular_terms=[['hr', 'ht', 'ht'], ['hr', 'hp', 'hp']],
            radial_multiplier=radial_multiplier,
            div_by_sin=[False, False],
            ang_factor=[-1.0, 1.0],
            rad_indexer=indexer,
        )
        # compute the total term
        total = _sym1 + _sym2 + _sym3 + _sym4 + _sym5 + _sym6
        # compute and return the (dimensionalized) contribution
        return 0.5 * total * _dim_term

    # - Kappa(2)_ABC -> similar to Eq.(B3) in Lee (2012)
    def _kappa_2_abc(self, indexer: slice) -> float:
        """Internal utility method used to compute term Kappa^2_ABC (similar to Eq.(B3) in Lee, 2012)

        Parameters
        ----------
        indexer : np.s_
            Numpy slicing object used to retrieve a specific term of the coupling coefficient profile.

        Returns
        -------
        float
            The integration result.
        """
        # store factor in local variable to be used in the radial multiplier
        _gam = (self._Gamma_1 - 1.0) ** (2.0)
        # compute the radial multiplier term
        if self._use_polytropic:
            _radial_multiplier = self._P_normed * _gam
        else:
            # check if isotropic derivative attribute was set
            try:
                # isentropic Gamma_1 derivative attribute present
                _radial_multiplier = self._P_normed * (
                    _gam + (self._rho * self._gamma_1_adDer_profile)
                )
                # NOTE the flipped profile read-in (MESA vs. GYRE output)
            except AttributeError:
                # no isentropic Gamma_1 derivative attribute present
                _radial_multiplier = self._P_normed * _gam
                # issue a warning that you are neglecting this derivative
                logger.warning(
                    'No isentropic derivative attribute was found. Neglecting this term!'
                )
        # compute the dimensionalizing term
        _dim_term = self._norm_P * self._R_star ** (3.0)
        # compute the radial term
        rad_term = self._rad_int(
            ['_rad_diverg', '_rad_diverg', '_rad_diverg'],
            _radial_multiplier,
            indexer=indexer,
        )
        # compute the angular term
        # - see Eq. (B18) in Lee(2012)
        ang_term = HI.quadratic_angular_integral(
            my_obj=self,
            m_list_for_azimuthal_check=self._m_list,
            conj=self._conj_cc,
            descrips=self._map_hough_to_mode(['hr', 'hr', 'hr'], [1, 2, 3]),
            div_by_sin=False,
            mul_with_mu=False,
            nr_omp_threads=self._omp_threads,
            use_parallel=self._use_parallel,
            num_integ_method=self._numerical_integration_method,
            use_cheby_integration=self._use_cheby,
            cheby_order_multiplier=self._cheby_order_mul,
            cheby_blow_up_protection=self._cheby_blow_up,
            cheby_blow_up_factor=self._cheby_blow_up_fac,
        )
        # compute the total contribution
        total = rad_term * ang_term
        # compute and return the contribution
        return total * 0.5 * _dim_term

    # - Kappa(3)_ABC -> similar to Eq.(B4) in Lee (2012)
    def _kappa_3_abc(self, indexer: slice) -> float:
        """Internal utility method used to compute term Kappa^3_ABC (similar to Eq.(B4) in Lee, 2012)

        Parameters
        ----------
        indexer : np.s_
            Numpy slicing object used to retrieve a specific term of the coupling coefficient profile.

        Returns
        -------
        float
            The integration result.
        """
        # compute the radial multiplier term
        _radial_multiplier = self._P_normed
        # compute the dimensionalizing term
        _dim_term = self._norm_P * self._R_star ** (3.0)
        # compute the symmetric terms
        _sym1 = self._symmetrization_s_term_integral(  # term 2 (& 1) in Eq.(B19) of Lee (2012)
            radial_terms=['_rad_der_x_z_1', '_z_1', '_rad_der_x_z_2_c_1_omega'],
            angular_terms=[
                ['hr', 'theta_der_hr_theta', 'ht'],
                ['hr', 'mhr', 'hp'],
            ],
            radial_multiplier=_radial_multiplier,
            div_by_sin=[False, True],
            ang_factor=[1.0, -1.0],
            rad_indexer=indexer,
        )
        _sym2 = self._symmetrization_s_term_integral(  # term 1 in Eq.(B19) of Lee (2012)
            radial_terms=[
                '_rad_der_x_z_1',
                '_z_2_over_c_1_omega',
                '_rad_der_x_z_2_c_1_omega',
            ],
            angular_terms=[['hr', 'ht', 'ht'], ['hr', 'hp', 'hp']],
            radial_multiplier=_radial_multiplier,
            div_by_sin=[False, False],
            ang_factor=[-1.0, 1.0],
            rad_indexer=indexer,
        )
        _sym3 = self._symmetrization_s_term_integral(  # term 5 in Eq.(B19) of Lee (2012)
            radial_terms=['_z_1', '_z_1', '_rad_der_x_z_2_c_1_omega'],
            angular_terms=[
                ['theta_der_hr_theta', 'hr', 'ht'],
                ['mhr', 'hr', 'hp'],
            ],
            radial_multiplier=_radial_multiplier,
            rad_indexer=indexer,
            div_by_sin=[False, True],
            ang_factor=[1.0, -1.0],
        )
        _sym4 = self._symmetrization_s_term_integral(  # term 3 in Eq.(B19) of Lee (2012)
            radial_terms=[
                '_z_1',
                '_z_2_over_c_1_omega',
                '_rad_der_x_z_2_c_1_omega',
            ],
            angular_terms=[
                ['hr', 'ht', 'ht'],
                ['theta_der_hr_theta', 'theta_der_ht_theta', 'ht'],
                ['hr', 'hp', 'hp'],
                ['mhr', 'phi_phi', 'hp'],
                ['mhr', 'theta_der_hp_theta', 'ht'],
                ['theta_der_hr_theta', 'theta_phi', 'hp'],
            ],
            radial_multiplier=_radial_multiplier,
            div_by_sin=[False, False, False, True, True, True],
            ang_factor=[-1.0, 1.0, 1.0, -1.0, -1.0, -1.0],
            rad_indexer=indexer,
        )
        _sym5 = self._symmetrization_s_term_integral(  # term 4 in Eq.(B19) of Lee (2012)
            radial_terms=[
                '_z_2_over_c_1_omega',
                '_z_2_over_c_1_omega',
                '_rad_der_x_z_2_c_1_omega',
            ],
            angular_terms=[
                ['ht', 'theta_der_ht_theta', 'ht'],
                ['hp', 'phi_phi', 'hp'],
                ['hp', 'theta_der_hp_theta', 'ht'],
                ['theta_phi', 'ht', 'hp'],
            ],
            radial_multiplier=_radial_multiplier,
            rad_indexer=indexer,
            div_by_sin=[False, False, False, True],
            ang_factor=[-1.0, 1.0, 1.0, 1.0],
        )
        _sym6 = self._symmetrization_s_term_integral(  # adjusted from terms 7 & 8 in Eq.(B19) of Lee (2012)
            radial_terms=['_z_1', '_z_2_over_c_1_omega', '_z_2_over_c_1_omega'],
            angular_terms=[
                ['hr', 'theta_phi', 'theta_der_hp_theta'],
                ['hr', 'phi_phi', 'phi_phi'],
                ['hr', 'theta_der_ht_theta', 'theta_der_ht_theta'],
            ],
            div_by_sin=[True, False, False],
            ang_factor=[-2.0, 1.0, 1.0],
            radial_multiplier=_radial_multiplier,
            rad_indexer=indexer,
        )
        _sym7 = self._symmetrization_s_term_integral(  # term 6 in Eq.(B19) of Lee (2012)
            radial_terms=[
                '_z_2_over_c_1_omega',
                '_z_2_over_c_1_omega',
                '_z_2_over_c_1_omega',
            ],
            angular_terms=[
                ['theta_der_ht_theta', 'theta_phi', 'theta_der_hp_theta'],
                ['theta_der_hp_theta', 'theta_phi', 'phi_phi'],
            ],
            div_by_sin=[True, True],
            ang_factor=[-1.0, -1.0],
            radial_multiplier=_radial_multiplier,
            rad_indexer=indexer,
        )
        _sym8 = self._symmetrization_s_term_integral(  # adjusted from term 9 in Eq.(B19) of Lee (2012)
            radial_terms=['_z_1', '_z_1', '_z_2_over_c_1_omega'],
            angular_terms=[
                ['hr', 'hr', 'theta_der_ht_theta'],
                ['hr', 'hr', 'phi_phi'],
            ],
            div_by_sin=[False, False],
            radial_multiplier=_radial_multiplier,
            ang_factor=[1.0, 1.0],
            rad_indexer=indexer,
        )
        # compute the total contribution for the symmetric terms
        _symmetric = (
            _sym1 + _sym2 + _sym3 + _sym4 + _sym5 + _sym6 + _sym7 + _sym8
        )
        # compute the non-symmetric terms
        _rad_1 = (
            2.0
            * self._rad_int(  # adjusted from term 10 in Eq.(B19) of Lee (2012)
                ['_rad_der_x_z_1', '_rad_der_x_z_1', '_rad_der_x_z_1'],
                multiplying_term=_radial_multiplier,
                indexer=indexer,
            )
        )
        _rad_2 = (
            4.0
            * self._rad_int(  # adjusted from term 10 in Eq.(B19) of Lee (2012)
                ['_z_1', '_z_1', '_z_1'],
                multiplying_term=_radial_multiplier,
                indexer=indexer,
            )
        )
        _ang_1_2 = HI.quadratic_angular_integral(
            my_obj=self,
            m_list_for_azimuthal_check=self._m_list,
            conj=self._conj_cc,
            descrips=self._map_hough_to_mode(['hr', 'hr', 'hr'], [1, 2, 3]),
            mul_with_mu=False,
            div_by_sin=False,
            nr_omp_threads=self._omp_threads,
            use_parallel=self._use_parallel,
            num_integ_method=self._numerical_integration_method,
            use_cheby_integration=self._use_cheby,
            cheby_order_multiplier=self._cheby_order_mul,
            cheby_blow_up_protection=self._cheby_blow_up,
            cheby_blow_up_factor=self._cheby_blow_up_fac,
        )
        _rad_3_4 = (
            2.0
            * self._rad_int(  # adjusted from term 11 in Eq.(B19) of Lee (2012)
                [
                    '_z_2_over_c_1_omega',
                    '_z_2_over_c_1_omega',
                    '_z_2_over_c_1_omega',
                ],
                multiplying_term=_radial_multiplier,
                indexer=indexer,
            )
        )
        _ang_3 = HI.quadratic_angular_integral(
            my_obj=self,
            m_list_for_azimuthal_check=self._m_list,
            conj=self._conj_cc,
            descrips=self._map_hough_to_mode(
                ['phi_phi', 'phi_phi', 'phi_phi'], [1, 2, 3]
            ),
            mul_with_mu=False,
            div_by_sin=False,
            nr_omp_threads=self._omp_threads,
            use_parallel=self._use_parallel,
            num_integ_method=self._numerical_integration_method,
            use_cheby_integration=self._use_cheby,
            cheby_order_multiplier=self._cheby_order_mul,
            cheby_blow_up_protection=self._cheby_blow_up,
            cheby_blow_up_factor=self._cheby_blow_up_fac,
        )
        _ang_4 = HI.quadratic_angular_integral(
            my_obj=self,
            m_list_for_azimuthal_check=self._m_list,
            conj=self._conj_cc,
            descrips=self._map_hough_to_mode(
                [
                    'theta_der_ht_theta',
                    'theta_der_ht_theta',
                    'theta_der_ht_theta',
                ],
                [1, 2, 3],
            ),
            div_by_sin=False,
            mul_with_mu=False,
            nr_omp_threads=self._omp_threads,
            use_parallel=self._use_parallel,
            num_integ_method=self._numerical_integration_method,
            use_cheby_integration=self._use_cheby,
            cheby_order_multiplier=self._cheby_order_mul,
            cheby_blow_up_protection=self._cheby_blow_up,
            cheby_blow_up_factor=self._cheby_blow_up_fac,
        )
        # compute the total contribution for the non-symmetric terms
        _non_symmetric = (_ang_1_2 * (_rad_1 + _rad_2)) + (
            _rad_3_4 * (_ang_3 + _ang_4)
        )
        # compute the total contribution
        total = _non_symmetric + _symmetric
        # return the total contribution of this term
        return 0.5 * total * _dim_term

    # - Kappa(4)_ABC -> similar to Eq.(B5) in Lee (2012)
    def _kappa_4_abc(self, indexer: slice) -> float:
        """Internal utility method used to compute term Kappa^4_ABC (similar to Eq.(B5) in Lee, 2012)

        Parameters
        ----------
        indexer : np.s_
            Numpy slicing object used to retrieve a specific term of the coupling coefficient profile.

        Returns
        -------
        float
            The integration result.
        """
        # compute the radial multiplier term
        _radial_multiplier = self._rho_normed * self._x ** (3.0)
        # construct radius term
        _radius = self._x * self._R_star
        if TYPE_CHECKING:
            assert isinstance(self._dgravpot, np.ndarray)
            assert isinstance(self._dgravpot2, np.ndarray)
            assert isinstance(self._dgravpot3, np.ndarray)
        # compute the symmetric terms and their specific additional radial multiplier
        # - see Eq.(B20) in Lee(2012)
        _add_multiplier1 = ((-1.0 / _radius ** (2.0)) * self._dgravpot) + (
            (1.0 / _radius) * self._dgravpot2
        )
        _sym = 0.5 * self._symmetrization_s_term_integral(
            radial_terms=['_z_1', '_z_2_over_c_1_omega', '_z_2_over_c_1_omega'],
            angular_terms=[['hr', 'ht', 'ht'], ['hr', 'hp', 'hp']],
            radial_multiplier=_radial_multiplier * _add_multiplier1,
            div_by_sin=[False, False],
            ang_factor=[1.0, -1.0],
            rad_indexer=indexer,
        )
        # compute the non-symmetric term and its additional radial multiplier
        # - see Eq.(B20) in Lee(2012)
        _rad_not_sym = self._rad_int(
            ['_z_1', '_z_1', '_z_1'],
            indexer=indexer,
            multiplying_term=_radial_multiplier * self._dgravpot3,
        )
        _ang_not_sym = HI.quadratic_angular_integral(
            my_obj=self,
            m_list_for_azimuthal_check=self._m_list,
            conj=self._conj_cc,
            descrips=self._map_hough_to_mode(['hr', 'hr', 'hr'], [1, 2, 3]),
            div_by_sin=False,
            mul_with_mu=False,
            nr_omp_threads=self._omp_threads,
            use_parallel=self._use_parallel,
            num_integ_method=self._numerical_integration_method,
            use_cheby_integration=self._use_cheby,
            cheby_order_multiplier=self._cheby_order_mul,
            cheby_blow_up_protection=self._cheby_blow_up,
            cheby_blow_up_factor=self._cheby_blow_up_fac,
        )
        # compute the common dimensionalization term
        _common_dim_term = self._norm_rho * self._R_star ** (6.0)
        # compute the total contribution
        total = _sym + (_rad_not_sym * _ang_not_sym)
        # return the total contribution of this term
        return -0.5 * total * _common_dim_term

    # method that computes the adiabatic quadratic coupling coefficients for a rotating star
    def adiabatic_coupling_rot(self, indexer: slice | int = np.s_[:]) -> None:
        """Internal method that computes and stores the quadratic coupling coefficients defined in Lee(2012).

        Parameters
        ----------
        indexer : np.s_ or int, optional
            Integer that creates Numpy slicing object used to retrieve a specific term of the coupling coefficient profile, if set. Leave at default value to get the regular coupling coefficient; by default np.s_[:].
        """
        # get the indexing object if necessary
        if isinstance(indexer, int):
            indexer = np.s_[:indexer]
            # get specific boolean used to indicate a profile is being computed.
            _profile = True
        else:
            _profile = False
        # compute and store the adiabatic quadratic coupling coefficient by computing the sum of the four quadratic coupling coefficient terms (see e.g. Lee(2012))
        self._cc_term_1 = self._kappa_1_abc(indexer)
        self._cc_term_2 = self._kappa_2_abc(indexer)
        self._cc_term_3 = self._kappa_3_abc(indexer)
        self._cc_term_4 = self._kappa_4_abc(indexer)
        self._coupling_coefficient = (
            self._cc_term_1
            + self._cc_term_2
            + self._cc_term_3
            + self._cc_term_4
        )
        # log a message that the coupling coefficient was computed
        if not _profile:
            logger.debug('The quadratic coupling coefficient was computed.')
        # print the debug information, if requested
        if self._store_debug:
            self.print_all_debug_properties()

    # method that computes the adiabatic quadratic coupling coefficient profile for a rotating star
    def adiabatic_coupling_rot_profile(self, progress_bar: bool=False) -> None:
        """Computes and stores the adiabatic mode coupling profile."""
        # initialize the numpy arrays that will hold the coefficient profile as well as the individual contributions
        self._coupling_coefficient_integrated = np.zeros_like(
            self._x, dtype=np.complex128 if self._use_complex else np.float64
        )
        self._cc_term_1_integrated = np.zeros_like(
            self._coupling_coefficient_integrated
        )
        self._cc_term_2_integrated = np.zeros_like(
            self._coupling_coefficient_integrated
        )
        self._cc_term_3_integrated = np.zeros_like(
            self._coupling_coefficient_integrated
        )
        self._cc_term_4_integrated = np.zeros_like(
            self._coupling_coefficient_integrated
        )
        # use that to compute the coupling coefficient profile --> lower edge case = NO profile --> set to zero
        if progress_bar:
            for _i in tqdm(range(2, self._x.shape[0] + 1), desc='Profile computations'):
                # compute the coupling coefficient profile parts
                self.adiabatic_coupling_rot(indexer=_i)
                # retrieve the value and store it
                if TYPE_CHECKING:
                    # these are numpy numbers!
                    assert isinstance(self.coupling_coefficient, np.ndarray)
                    assert isinstance(self._cc_term_1, np.ndarray)
                    assert isinstance(self._cc_term_2, np.ndarray)
                    assert isinstance(self._cc_term_3, np.ndarray)
                    assert isinstance(self._cc_term_4, np.ndarray)
                self._coupling_coefficient_integrated[
                    _i - 1
                ] = self.coupling_coefficient.copy()
                self._cc_term_1_integrated[_i - 1] = self._cc_term_1.copy()
                self._cc_term_2_integrated[_i - 1] = self._cc_term_2.copy()
                self._cc_term_3_integrated[_i - 1] = self._cc_term_3.copy()
                self._cc_term_4_integrated[_i - 1] = self._cc_term_4.copy()            
        else:
            for _i in range(2, self._x.shape[0] + 1):
                # compute the coupling coefficient profile parts
                self.adiabatic_coupling_rot(indexer=_i)
                # retrieve the value and store it
                if TYPE_CHECKING:
                    # these are numpy numbers!
                    assert isinstance(self.coupling_coefficient, np.ndarray)
                    assert isinstance(self._cc_term_1, np.ndarray)
                    assert isinstance(self._cc_term_2, np.ndarray)
                    assert isinstance(self._cc_term_3, np.ndarray)
                    assert isinstance(self._cc_term_4, np.ndarray)
                self._coupling_coefficient_integrated[
                    _i - 1
                ] = self.coupling_coefficient.copy()
                self._cc_term_1_integrated[_i - 1] = self._cc_term_1.copy()
                self._cc_term_2_integrated[_i - 1] = self._cc_term_2.copy()
                self._cc_term_3_integrated[_i - 1] = self._cc_term_3.copy()
                self._cc_term_4_integrated[_i - 1] = self._cc_term_4.copy()
        # print the debug information, if requested
        if self._store_debug:
            self.print_all_debug_properties()

    def _fill_hough_mode_debug_attributes(self) -> None:
        """Fills up the specific Hough debug attributes for the modes."""
        # fill up the debug attributes for each mode
        if self._store_debug:
            for _i in range(1, self._nr_modes + 1):
                self._fill_debug_attributes(
                    DmmQccR.get_mode_specific_attrs(mode_nr=_i)
                )

    def print_all_debug_properties(self):
        """Prints all debug properties in a nice format."""
        # verify debug properties have been stored
        if self._store_debug:
            # import a pretty printer
            import pprint

            # loop over the debug attributes/properties and print them
            _debug_dict = {
                x: y
                for (x, y) in self.__dict__.items()
                if ('max_min' in x) and ('_attr_list' not in x)
            }
            print('\n\n')
            pprint.pprint(_debug_dict)
            print('\n\n')
        else:
            logger.warning(
                'You are trying to print debug values, but did not activate the store_debug flag. No debug properties can be shown!'
            )
