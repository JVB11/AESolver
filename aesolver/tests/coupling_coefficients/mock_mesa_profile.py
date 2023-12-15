'''Python module containing functions that create a mock MESA profile file.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
'''
# import statements
import typing
import numpy as np
    
    
# store the MESA profile data
class MESAProfileData:
    """Contains the MESA profile data."""
    
    # attribute and type declarations
    # - body arrays
    gamma_1_adDer: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    h1: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    h2: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    he3: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    he4: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    c12: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    c13: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    n14: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    o16: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    o17: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    ne20: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    ne22: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    mg24: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    al27: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    si28: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    s32: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    ar36: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    brunt_N2: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    brunt_N2_structure_term: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    brunt_N2_composition_term: np.ndarray = \
        (np.ones((5,), dtype=np.float64), np.ndarray)
    lamb_S: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    lamb_S2: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    hyeq_lhs: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    hyeq_rhs: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    x: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    y: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    z: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    mlt_mixing_length: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    log_D_mix: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    log_D_conv: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    log_D_ovr: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    log_D_minimum: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    conv_mixing_type: np.ndarray = (7 * np.ones((5,), dtype=np.float64), np.ndarray)
    zone: np.ndarray = (np.linspace(1., 5., num=5, dtype=np.float64), np.ndarray)
    logT: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    logRho: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    luminosity: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    velocity: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    entropy: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    csound: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    mu: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    q: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    radius: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    tau: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    pressure: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    opacity: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    eps_nuc: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    non_nuc_neu: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    log_conv_vel: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    mass: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    mmid: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    grada: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    gradT: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    gradr: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    pressure_scale_height: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    grav: np.ndarray = (np.ones((5,), dtype=np.float64), np.ndarray)
    # - header information
    model_number: str = ('5157', str)
    num_zones: str = ('5', str)
    initial_mass: str = ('1.0', str)
    initial_z: str = ('0.02', str)
    star_age: str = ('2.0e7', str)
    time_step: str = ('0.1e5', str)
    Teff: str = ('1250.0e1', str)
    photosphere_L: str = ('2.3e3', str)
    photosphere_r: str = ('6.5', str)
    center_eta: str = ('-4.95', str)
    center_h1: str = ('9e-2', str)
    center_he3: str = ('7e-10', str)
    center_he4: str = ('9e-1', str)
    center_c12: str = ('1e-4', str)
    center_n14: str = ('8e-3', str)
    center_o16: str = ('3e-4', str)
    center_ne20: str = ('1.5e-3', str)
    star_mass_h1: str = ('3.3', str)
    star_mass_he3: str = ('4.e-4', str)
    star_mass_he4: str = ('2.6', str)
    star_mass_c12: str = ('5.4e-3', str)
    star_mass_n14: str = ('2e-2', str)
    star_mass_o16: str = ('2.8e-2', str)
    star_mass_ne20: str = ('1.e-2', str)
    he_core_mass: str = ('0.0', str)
    c_core_mass: str = ('0.0', str)
    o_core_mass: str = ('0.0', str)
    si_core_mass: str = ('0.0', str)
    fe_core_mass: str = ('0.0', str)
    neutron_rich_core_mass: str = ('0.0', str)
    tau10_mass: str = ('6.0', str)
    tau10_radius: str = ('6.5', str)
    tau100_mass: str = ('6.0', str)
    tau100_radius: str = ('6.5', str)
    dynamic_time: str = ('7.e4', str)
    kh_timescale: str = ('6.e4', str)
    nuc_timescale: str = ('2.6e7', str)
    power_nuc_burn: str = ('2.3e3', str)
    power_h_burn: str = ('2.3e3', str)
    power_he_burn: str = ('9.e-28', str)
    power_neu: str = ('1.6e2', str)
    burn_min1: str = ('5.e1', str)
    burn_min2: str = ('1.e3', str)
    time_seconds: str = ('1.9e15', str)
    version_number: str = ('"15140"', str)
    compiler: str = ('"compiler"', str)
    build: str = ('"10.2.0"', str)
    MESA_SDK_version: str = ('"x86_64-linux-21.4.1"', str)
    date: str = ('"20230119"', str)
    
    # class method used to obtain all attributes from this class
    # in a dictionary format
    @classmethod
    def get_dict(cls) -> dict[str, typing.Any]:
        """Retrieves all attributes of this class in a dictionary format.
        
        Returns
        -------
        dict[str, typing.Any]
            The dictionary containing all attributes of this class.
        """
        return {
            a: getattr(cls, a)[0]
            for a in vars(cls) if ('__' not in a) and ('get_dict' != a)
        }
