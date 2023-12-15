'''Python module containing the expected output/attributes for the 'QuadraticCouplingCoefficientRotating' class that is part of the aesolver.coupling_coefficients module.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
'''
# import statements
import typing
import numpy as np


# create template input arrays
template_double_arr_multi_cut = np.ones(
    (3,), dtype=np.float64
    )
template_double_arr_cut = np.ones(
    (4,), dtype=np.float64
    )
template_double_arr = np.ones(
    (5,), dtype=np.float64
    )
template_complex_arr_cut = np.ones(
    (4,), dtype=np.complex128
    )
template_complex_arr = np.ones(
    (5,), dtype=np.complex128
    )


# define the expected attributes common among modes
# after initialization
class ExpectedCommonAttrsAfterInitQCCR:
    """Defines the expected attribute values shared among modes after initialization."""
    
    M_star: np.float64 = np.array(
        [1.5e34], dtype=np.float64
        )[0]
    R_star: np.float64 = np.array(
        [1.0e12], dtype=np.float64
        )[0]
    L_star: np.float64 = np.array(
        [2.0e37], dtype=np.float64
        )[0]
    # Delta_p
    # Delta_g
    V_2: np.ndarray = template_double_arr_multi_cut
    As: np.ndarray = template_double_arr_multi_cut
    U: np.ndarray = template_double_arr_multi_cut
    c_1: np.ndarray = template_double_arr_multi_cut
    Gamma_1: np.ndarray = template_double_arr_multi_cut
    nabla: np.ndarray = template_double_arr
    nabla_ad: np.ndarray = template_double_arr
    dnabla_ad: np.ndarray = template_double_arr
    # upsilon_r
    # c_lum
    # c_rad
    # c_thn
    # c_thk
    # c_eps
    # kap_rho
    # kap_T
    # eps_rho
    # eps_T
    Omega_rot: np.ndarray = template_double_arr
    M_r: np.ndarray = np.linspace(
        0.0, 1.5e34, num=5, dtype=np.float64
        )[1:-1]
    P: np.ndarray = np.linspace(
        1.0e5, 1.0, num=5, dtype=np.float64
        )[1:-1]
    rho: np.ndarray = np.linspace(
        1.0e2, 1.0e-3, num=5, dtype=np.float64
        )[1:-1]
    T: np.ndarray = np.linspace(
        1.0e8, 1.0e3, num=5, dtype=np.float64
        )
    x: np.ndarray = np.linspace(
        0.0, 1.0, num=5, dtype=np.float64
        )[1:-1]
    
    # retrieve the output attributes in a dictionary
    @classmethod
    def get_dict(cls) -> dict[str, typing.Any]:
        """Retrieves the input arguments in dictionary format."""
        return {
            _k: getattr(cls, _k) for _k in vars(cls)
            if ('__' not in _k) and ('get_dict' != _k)
            }


# define the expected mode-specific attributes
# after initialization
class ExpectedModeSpecificAttrsAfterInitQCCR:
    """Defines the expected mode-specific attribute values after initialization."""
    
    # attribute declarations (mode-specific!)
    n: np.int32 = np.array(
        [5, 5, 5], dtype=np.int32
        )
    omega: np.void = np.array(
        [0.68, 0.67, 0.65], dtype=np.complex128
        )
    y_1: np.ndarray = np.array(
        [np.linspace(
        1.0, 2.0, num=5, dtype=np.complex128
        )[1:-1]]*3
        )
    y_2: np.ndarray = np.array(
        [ np.linspace(
        0.5, 1.5, num=5, dtype=np.complex128
        )[1:-1]]*3
        )
    y_3: np.ndarray = np.array(
        [template_complex_arr]*3  # WHY IS Y_3 not cut?
        )
    y_4: np.ndarray = np.array(
        [template_complex_arr]*3  # WHY IS Y_4 not cut?
        ) 
    l: np.int32 = np.array(
        [2, 2, 2], dtype=np.int32
        )
    m: np.int32 = np.array(
        [2, 2, 2], dtype=np.int32
        )
    l_i: np.void = np.array(
        [1.8, 1.8, 1.8], dtype=np.complex128
        ) 
    n_p: np.int32 = np.array(
        [0, 0, 0], dtype=np.int32
        )
    n_g: np.int32 = np.array(
        [23, 24, 25], dtype=np.int32
        )
    n_pg: np.int32 = np.array(
        [-23, -24, -25], dtype=np.int32
        )
    xi_r: np.ndarray = np.array(
        [np.array([0., 1.5, 1., 0.5, 1.25],
                  dtype=np.complex128)[1:-1]]*3
        )
    xi_h: np.ndarray = np.array(
        [np.array([0., 1.5, 1., 0.5, 1.25],
                  dtype=np.complex128)[1:-1]]*3
        )
    eul_phi: np.ndarray = np.array(
        [template_complex_arr_cut[:-1]]*3
        )
    deul_phi: np.ndarray = np.array(
        [template_complex_arr_cut[:-1]]*3
        )
    lag_L: np.ndarray = np.array(
        [template_complex_arr]*3
        )
    eul_P: np.ndarray = np.array(
        [template_complex_arr]*3
        )
    eul_rho: np.ndarray = np.array(
        [template_complex_arr]*3
        )
    eul_T: np.ndarray = np.array(
        [template_complex_arr]*3
        )
    lag_P: np.ndarray = np.array(
        [template_complex_arr]*3
        )
    lag_rho: np.ndarray = np.array(
        [template_complex_arr]*3
        )
    lag_T: np.ndarray = np.array(
        [template_complex_arr]*3
        )
    my_lambda: np.ndarray = np.array(
        [[2.0+0.0j], [3.0+0.0j], [4.0+0.0j]],
        dtype=np.complex128
        )
    # dimensionalized
    xi_r_dim: np.ndarray = np.array(  # WHY ARE THESE NOT CUT?
        [1.e12*np.array([0., 1.5, 1., 0.5, 1.25],
                        dtype=np.complex128)]*3
        )
    xi_h_dim: np.ndarray = np.array(
        [1.e12*np.array([0., 1.5, 1., 0.5, 1.25],
                        dtype=np.complex128)]*3
        )
    eul_phi_dim: np.ndarray = np.array(
        [1.001145e15 * template_complex_arr]*3
        )
    deul_phi_dim: np.ndarray = np.array(
        [1001.145 * template_complex_arr]*3
        )
    # lag_L_dim: np.ndarray = np.array(
    #     [2.0e37 * template_complex_arr]*3
    #     )
    eul_P_dim: np.ndarray = np.array(
        [template_complex_arr]*3
        )
    eul_rho_dim: np.ndarray = np.array(
        [template_complex_arr]*3
        )
    eul_T_dim: np.ndarray = np.array(
        [template_complex_arr]*3
        )
    lag_P_dim: np.ndarray = np.array(
        [template_complex_arr]*3
        )
    lag_rho_dim: np.ndarray = np.array(
        [template_complex_arr]*3
        )
    lag_T_dim: np.ndarray = np.array(
        [template_complex_arr]*3
        )
    # this is also not mode-specific!
    Omega_rot_dim: np.ndarray = np.array(
        [3.16408755e-5 * template_double_arr]*3
        )
    # THIS SHOULD NOT BE MODE-SPECIFIC?
    G: np.ndarray = np.array(
        [6.674299999999999e-8]*3,
        dtype=np.float64
        )
    
    # retrieve the output attributes in a dictionary
    @classmethod
    def get_dict(cls, mode_nr=1) -> dict[str, typing.Any]:
        """Retrieves the input arguments in  dictionary format.
        
        Parameters
        ----------
        mode_nr : int, optional
            The number of the mode for which you are trying to access expected values; by default 1.
            
        Returns
        -------
        dict[str, typing.Any]
            Contains the expected attributes for the requested mode.
        """
        return {
            "lambda" if _k == 'my_lambda' else _k:
                getattr(cls, _k)[mode_nr - 1]
            for _k in vars(cls)
            if ('__' not in _k) and ('get_dict' != _k)
            }


# define the expected MESA profile attributes
# after initialization
class ExpectedProfileAttrsAfterInitQCCR:
    """Defines the expected profile attribute values after initialization."""
    
    # PROFILES
    gamma_1_adDer: np.ndarray = \
        template_double_arr_multi_cut[::-1]
    h1: np.ndarray = template_double_arr[::-1]
    h2: np.ndarray = template_double_arr[::-1]
    he3: np.ndarray = template_double_arr[::-1]
    he4: np.ndarray = template_double_arr[::-1]
    c12: np.ndarray = template_double_arr[::-1]
    c13: np.ndarray = template_double_arr[::-1]
    n14: np.ndarray = template_double_arr[::-1]
    o16: np.ndarray = template_double_arr[::-1]
    o17: np.ndarray = template_double_arr[::-1]
    ne20: np.ndarray = template_double_arr[::-1]
    ne22: np.ndarray = template_double_arr[::-1]
    mg24: np.ndarray = template_double_arr[::-1]
    al27: np.ndarray = template_double_arr[::-1]
    si28: np.ndarray = template_double_arr[::-1]
    s32: np.ndarray = template_double_arr[::-1]
    ar36: np.ndarray = template_double_arr[::-1]
    brunt_N2: np.ndarray = \
        template_double_arr_multi_cut[::-1]
    brunt_N2_structure_term: np.ndarray = \
        template_double_arr[::-1]
    brunt_N2_composition_term: np.ndarray = \
        template_double_arr[::-1]
    lamb_S: np.ndarray = template_double_arr[::-1]
    lamb_S2: np.ndarray = template_double_arr[::-1]
    hyeq_lhs: np.ndarray = \
        template_double_arr[::-1]
    hyeq_rhs: np.ndarray = \
        template_double_arr[::-1]
    x: np.ndarray = template_double_arr[::-1]
    y: np.ndarray = template_double_arr[::-1]
    z: np.ndarray = template_double_arr[::-1]
    mlt_mixing_length: np.ndarray = \
        template_double_arr[::-1]
    log_D_mix: np.ndarray = \
        template_double_arr[::-1]
    log_D_conv: np.ndarray = \
        template_double_arr[::-1]
    log_D_ovr: np.ndarray = \
        template_double_arr[::-1]
    log_D_minimum: np.ndarray = \
        template_double_arr[::-1]
    conv_mixing_type: np.ndarray = \
        7.0 * template_double_arr[::-1]
    zone: np.ndarray = \
        np.linspace(1., 5., num=5, dtype=np.float64)[::-1]
    logT: np.ndarray = template_double_arr[::-1]
    logRho: np.ndarray = template_double_arr[::-1]
    luminosity: np.ndarray = \
        template_double_arr[::-1]
    velocity: np.ndarray = \
        template_double_arr[::-1]
    entropy: np.ndarray = \
        template_double_arr[::-1]
    csound: np.ndarray = \
        template_double_arr[::-1]
    mu: np.ndarray = template_double_arr[::-1]
    q: np.ndarray = template_double_arr[::-1]
    radius: np.ndarray = \
        template_double_arr[::-1]
    tau: np.ndarray = template_double_arr[::-1]
    pressure: np.ndarray = \
        template_double_arr[::-1]
    opacity: np.ndarray = \
        template_double_arr[::-1]
    eps_nuc: np.ndarray = \
        template_double_arr[::-1]
    non_nuc_neu: np.ndarray = \
        template_double_arr[::-1]
    log_conv_vel: np.ndarray = \
        template_double_arr[::-1]
    mass: np.ndarray = template_double_arr[::-1]
    mmid: np.ndarray = template_double_arr[::-1]
    grada: np.ndarray = \
        template_double_arr[::-1]
    gradT: np.ndarray = \
        template_double_arr[::-1]
    gradr: np.ndarray = \
        template_double_arr[::-1]
    pressure_scale_height: np.ndarray = \
        template_double_arr[::-1]
    grav: np.ndarray = template_double_arr[::-1]
    # HEADER
    model_number: str = '5157'
    num_zones: str = '5'
    initial_mass: str = '1.0'
    initial_z: str = '0.02'
    star_age: str = '2.0e7'
    time_step: str = '0.1e5'
    Teff: str = '1250.0e1'
    photosphere_L: str = '2.3e3'
    photosphere_r: str = '6.5'
    center_eta: str = '-4.95'
    center_h1: str = '9e-2'
    center_he3: str = '7e-10'
    center_he4: str = '9e-1'
    center_c12: str = '1e-4'
    center_n14: str = '8e-3'
    center_o16: str = '3e-4'
    center_ne20: str = '1.5e-3'
    star_mass_h1: str = '3.3'
    star_mass_he3: str = '4.e-4'
    star_mass_he4: str = '2.6'
    star_mass_c12: str = '5.4e-3'
    star_mass_n14: str = '2e-2'
    star_mass_o16: str = '2.8e-2'
    star_mass_ne20: str = '1.e-2'
    he_core_mass: str = '0.0'
    c_core_mass: str = '0.0'
    o_core_mass: str = '0.0'
    si_core_mass: str = '0.0'
    fe_core_mass: str = '0.0'
    neutron_rich_core_mass: str = '0.0'
    tau10_mass: str = '6.0'
    tau10_radius: str = '6.5'
    tau100_mass: str = '6.0'
    tau100_radius: str = '6.5'
    dynamic_time: str = '7.e4'
    kh_timescale: str = '6.e4'
    nuc_timescale: str = '2.6e7'
    power_nuc_burn: str = '2.3e3'
    power_h_burn: str = '2.3e3'
    power_he_burn: str = '9.e-28'
    power_neu: str = '1.6e2'
    burn_min1: str = '5.e1'
    burn_min2: str = '1.e3'
    time_seconds: str = '1.9e15'
    version_number: str = '"15140"'
    compiler: str = '"compiler"'
    build: str = '"10.2.0"'
    MESA_SDK_version: str = '"x86_64-linux-21.4.1"'
    date: str = '"20230119"'    

    # retrieve the MESA profile attributes in a dictionary
    @classmethod
    def get_dict(cls) -> dict[str, typing.Any]:
        """Retrieves the MESA profile attributes in  dictionary format.
            
        Returns
        -------
        dict[str, typing.Any]
            Contains the expected attributes.
        """
        return {
            f"_{_k}_profile": getattr(cls, _k)
            for _k in vars(cls)
            if ('__' not in _k) and ('get_dict' != _k)
            }


# define the expected computed attributes
# after initialization
class ExpectedComputedAttrsAfterInitQCCR:
    """Defines the expected computed attribute values after initialization."""
    
    # attribute definitions
    _dgravpot: np.ndarray = np.array(
        [4004.58, 2002.29, 1334.86, 1001.145],
        dtype=np.float64
        )
    _dgravpot2: np.ndarray = np.array(
        [ 6.28719686e-05,  4.19282739e-05, 
         2.09650013e-05, -1.16357273e-09],
        dtype=np.float64
        )
    # 8.38717274e-7 = APPROX. factor dgravpot2
    _dgravpot3: np.ndarray = np.array(
        [-2.51919829e-04, -8.39895444e-05,
         -2.80058016e-05, -8.40515363e-10],
        dtype=np.float64
        )
    # -1.25663706e25 = APPROX. factor dgravpot3
    _drho: np.ndarray = np.array(
        [-3.00363230e2, -1.00140473e2,
         -3.33912303e1, -1.00214386e-3],
        dtype=np.float64
        )
    _drho2: np.ndarray = np.array(
        [1.06783806e-09, 5.33943999e-10,
         2.00278941e-10, 6.68428836e-11],
        dtype=np.float64
        )
    _drho3: np.ndarray = np.array(
        [-2.53603423e-21, -1.73511823e-21,
         -9.34202231e-22, -1.33286231e-22],
        dtype=np.float64
        )
    _dP: np.ndarray = np.array(
        [-3.00344501e+05, -1.00115501e+05,
         -3.33725011e+04, -1.00114500e+00],
        dtype=np.float64
        )
    _dP2: np.ndarray = np.array(
        [1.20282858e+06, 2.00510265e+05,
         4.45726171e+04, 1.00329131e+00],
        dtype=np.float64
        )
    _dP3: np.ndarray = np.array(
        [ 5.66586290e-02,  1.25958844e-02, 
         2.09999307e-03, -6.69209103e-08],
        dtype=np.float64
        )
    # normed attribute definitions
    _P_normed: np.ndarray = np.array(
        [1.0, 6.66671111e-01, 3.33342222e-1, 1.33332889e-05],
        dtype=np.float64
        )
    _dP_normed: np.ndarray = np.array(
        [-4.00458e+00, -1.3348689e+00,
         -4.44965199e-01, -1.33485555e-05],
        dtype=np.float64
        )
    _dP2_normed: np.ndarray = np.array(
        [1.60376609e+01, 2.67346129e+00,
         5.94299580e-01, 1.33771729e-05],
        dtype=np.float64
        )
    _dP3_normed: np.ndarray = np.array(
        [ 7.55445869e-07,  1.67944566e-07, 
         2.79998143e-08, -8.92275830e-13],
        dtype=np.float64
        )
    _rho_normed: np.ndarray = np.array(
        [1.0, 6.66671111e-01, 3.33342222e-01, 1.33332889e-05],
        dtype=np.float64
        )
    _drho_normed: np.ndarray = np.array(
        [-4.00482971e+00, -1.33520185e+00,
         -4.45214919e-01, -1.33618735e-05],
        dtype=np.float64
        )
    _drho2_normed: np.ndarray = np.array(
        [1.42377933e-11, 7.11922959e-12,
         2.67037698e-12, 8.91235478e-13],
        dtype=np.float64
        )
    _drho3_normed: np.ndarray = np.array(
        [-3.38136770e-23, -2.31348326e-23,
         -1.24559882e-23, -1.77714382e-24],
        dtype=np.float64
        )
    
    # retrieve the computed attributes in a dictionary
    @classmethod
    def get_dict(cls) -> dict[str, typing.Any]:
        """Retrieves the attributes computed during initialization in dictionary format.
            
        Returns
        -------
        dict[str, typing.Any]
            Contains the expected attributes.
        """
        return {
            _k: getattr(cls, _k)
            for _k in vars(cls)
            if ('__' not in _k) and ('get_dict' != _k)
            }   


# define the expected radial vector attributes
# after initialization
class ExpectedRadialVectorAttrsAfterInitQCCR:
    """Defines the expected radial vector attribute values after initialization."""
    
    z_1: np.ndarray = np.array(
        [np.array([4.43113463+0.j, 5.31736155+0.j,
                   6.20358848+0.j, 7.0898154 +0.j])]*3
        )
    z_2: np.ndarray = np.array(
        [np.array([2.65868078+0.j, 3.5449077 +0.j,
                   4.43113463+0.j, 5.31736155+0.j])]*3
        )    
    x_z_1: np.ndarray = np.array(
        [np.array([1.10778366+0.j, 2.65868078+0.j,
                   4.65269136+0.j, 7.0898154 +0.j])]*3
        )
    z_2_over_c_1_omega: np.ndarray = np.array(
        [np.array([0.04185884+0.j, 0.05581178+0.j,
                   0.06976473+0.j, 0.08371767+0.j]),
         np.array([0.08266114+0.j, 0.11021485+0.j,
                   0.13776856+0.j, 0.16532227+0.j]),
         np.array([0.23369482+0.j, 0.31159309+0.j,
                   0.38949137+0.j, 0.46738964+0.j])]
        )
    x_z_2_over_c_1_omega: np.ndarray = np.array(
        [np.array([0.01046471+0.j, 0.02790589+0.j,
                   0.05232354+0.j, 0.08371767+0.j]),
         np.array([0.02066528+0.j, 0.05510742+0.j,
                   0.10332642+0.j, 0.16532227+0.j]),
         np.array([0.05842371+0.j, 0.15579655+0.j,
                   0.29211853+0.j, 0.46738964+0.j])]
        )

    # retrieve the computed attributes in a dictionary
    @classmethod
    def get_dict(cls, mode_nr: int=1) -> dict[str, typing.Any]:
        """Retrieves the attributes computed during initialization in dictionary format.
        
        Parameters
        ----------
        mode_nr : int, optional
            The number of the mode for which the radial vectors are being checked; by default 1.
            
        Returns
        -------
        dict[str, typing.Any]
            Contains the expected attributes.
        """
        return {
            _k: getattr(cls, _k)[mode_nr - 1]
            for _k in vars(cls)
            if ('__' not in _k) and ('get_dict' != _k)
            }      
    
    
# define the expected radial vector gradient attributes
# after initialization
class ExpectedRadialVectorGradientAttrsAfterInitQCCR:
    """Defines the expected radial vector gradient attribute values after initialization."""
    
    rad_der_x_z_1: np.ndarray = np.array(
        [np.array([-8.66777322+0.j, -10.07998608+0.j,
                   -11.27064221+0.j,
                   -12.23974161+0.j]),
         np.array([-8.50350748+0.j, -9.8609651 +0.j,
                   -10.99686599+0.j,
                   -11.91121014+0.j]),
         np.array([-7.81671161+0.j, -8.94523727+0.j,
                   -9.8522062 +0.j,
                   -10.53761839+0.j])]
        )
    rad_der_x_z_2_c_1_omega: np.ndarray = np.array(
        [np.array([2634.39649019+0.j, 2343.03962136+0.j,
                   1467.2289944 +0.j,
                   6.96460932+0.j]),
         np.array([5197.98033411+0.j, 4621.75671705+0.j,
                   2891.37746018+0.j,
                   6.8425635+0.j]),
         np.array([1.46873355e+04+0.j, 1.30566499e+04+0.j,
                   8.16300165e+03+0.j,
                   6.39079901e+00+0.j])]
        )

    # retrieve the computed attributes in a dictionary
    @classmethod
    def get_dict(cls, mode_nr: int=1) -> dict[str, typing.Any]:
        """Retrieves the attributes computed during initialization in dictionary format.
        
        Parameters
        ----------
        mode_nr : int, optional
            The number of the mode for which the radial vectors are being checked; by default 1.
            
        Returns
        -------
        dict[str, typing.Any]
            Contains the expected attributes.
        """
        return {
            _k: getattr(cls, _k)[mode_nr - 1]
            for _k in vars(cls)
            if ('__' not in _k) and ('get_dict' != _k)
            }     


# define the expected radial divergence attribute
# after initialization
class ExpectedRadialDivergenceAttrAfterInitQCCR:
    """Defines the expected radial divergence attribute values after initialization."""
    
    rad_diverg: np.ndarray = np.array(
        [np.array([0.11077837+0.j, 0.44311346+0.j,
                   0.99700529+0.j, 1.77245385+0.j])]*3
        )

    # retrieve the computed attributes in a dictionary
    @classmethod
    def get_dict(cls, mode_nr: int=1) -> dict[str, typing.Any]:
        """Retrieves the attributes computed during initialization in dictionary format.
        
        Parameters
        ----------
        mode_nr : int, optional
            The number of the mode for which the radial vectors are being checked; by default 1.
            
        Returns
        -------
        dict[str, typing.Any]
            Contains the expected attributes.
        """
        return {
            _k: getattr(cls, _k)[mode_nr - 1]
            for _k in vars(cls)
            if ('__' not in _k) and ('get_dict' != _k)
            }    
