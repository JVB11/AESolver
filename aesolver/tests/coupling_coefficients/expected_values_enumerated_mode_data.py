'''Python module that contains the expected output for calls to methods of the enumeration class 'ModeData' that is part of the aesolver.coupling_coefficients module.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
'''


def expected_vals_mode_independent_names() -> list[str]:
    """Retrieves the expected values of the method 'retrieve_mode_independent_names'.

    Returns
    -------
    list[str]
        Contains the expected values of the method that will be tested.
    """
    return [
        'M_star', 'R_star', 'L_star', 'Delta_p',
        'Delta_g', 'V_2', 'As', 'U', 'c_1', 'Gamma_1',
        'nabla', 'nabla_ad', 'dnabla_ad', 'upsilon_r',
        'c_lum', 'c_rad', 'c_thn', 'c_thk', 'c_eps',
        'kap_rho', 'kap_T', 'eps_rho', 'eps_T',
        'Omega_rot', 'M_r', 'P', 'rho', 'T', 'x'
        ]
