'''Python module containing functions that create a dictionary that mocks GYRE detail files.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
'''
# import modules
import typing
import numpy as np


# create a custom void/complex data type
void_type = np.dtype(
        [('re', np.float64), ('im', np.float64)]
        )
# create template input arrays
template_double_arr = np.ones(
    (5,), dtype=np.float64
    )
template_void_arr = np.array(
    [(1., 0.), (1., 0.), (1., 0.), (1., 0.), (1., 0.)],
    dtype=void_type
    )   
template_complex_arr = np.ones(
    (5,), dtype=np.complex128
    )


# create a object that returns the expected values for each parameter
# in the GYRE dictionaries
class GYREDetailDictData:
    """Holds the expected values for GYRE Detail file dictionaries."""
    
    # attribute type and value declarations
    # -- original input
    n: np.int32 = np.array([5,5,5], np.int32)  # grid size!
    omega: np.void = np.array(
        [0.68, 0.67, 0.65], dtype=np.complex128
        )
    x: np.ndarray = np.linspace(
        0.0, 1.0, num=5, dtype=np.float64
        )
    y_1: np.ndarray = np.linspace(
        1.0, 2.0, num=5, dtype=np.complex128
        )
    y_2: np.ndarray = np.linspace(
        0.5, 1.5, num=5, dtype=np.complex128
        )
    y_3: np.ndarray = template_complex_arr
    y_4: np.ndarray = template_complex_arr
    freq: np.void = np.array(
        [2.0, 3.0, 4.0], dtype=np.complex128
        )
    freq_complex: np.void = np.array(
        [2.0+0.1j, 3.0-0.25j, 4.0-0.5j],
        dtype=np.complex128
        )
    freq_units: np.bytes_ = np.array(
        [b'CYC_PER_DAY']
        )[0]
    freq_frame: np.bytes_ = np.array(
        [b'INERTIAL']
        )[0]
    l: np.int32 = np.array([2], np.int32)[0]
    m: np.int32 = np.array([2], np.int32)[0]
    l_i: np.void = np.array(
        [1.8], dtype=np.complex128
        )[0]
    n_p: np.int32 = np.array([0], np.int32)[0]
    n_g: np.int32 = np.array([23,24,25], np.int32)
    n_pg: np.int32 = np.array([-23,-24,-25], np.int32)
    # TODO: also convert this set of void arrays to complex arrays???
    xi_r: np.ndarray = np.array(
        [(0., 0.), (1.5, 0.), (1., 0.), (0.5, 0.), (1.25, 0.)],
        dtype=void_type
        )
    xi_h: np.ndarray = np.array(
        [(0., 0.), (1.5, 0.), (1., 0.), (0.5, 0.), (1.25, 0.)],
        dtype=void_type
        )
    eul_phi: np.ndarray = template_void_arr
    deul_phi: np.ndarray = template_void_arr
    lag_L: np.ndarray = template_void_arr
    eul_P: np.ndarray = template_void_arr
    eul_rho: np.ndarray = template_void_arr
    eul_T: np.ndarray = template_void_arr
    lag_P: np.ndarray = template_void_arr
    lag_rho: np.ndarray = template_void_arr
    lag_T: np.ndarray = template_void_arr
    my_lambda: np.ndarray = np.array(
        [2.0+0.0j, 3.0+0.0j, 4.0+0.0j],
        dtype=np.complex128
        )
    M_star: np.float64 = np.array(
        [1.5e34], dtype=np.float64
        )[0]
    R_star: np.float64 = np.array(
        [1.0e12], dtype=np.float64
        )[0]
    L_star: np.float64 = np.array(
        [2.0e37], dtype=np.float64
        )[0]
    V_2: np.ndarray = template_double_arr
    As: np.ndarray = template_double_arr
    U: np.ndarray = template_double_arr
    c_1: np.ndarray = template_double_arr
    Gamma_1: np.ndarray = template_double_arr
    nabla: np.ndarray = template_double_arr
    nabla_ad: np.ndarray = template_double_arr
    dnabla_ad: np.ndarray = template_double_arr
    Omega_rot: np.ndarray = template_double_arr
    M_r: np.ndarray = np.linspace(
        0.0, 1.5e34, num=5, dtype=np.float64
        )
    P: np.ndarray = np.linspace(
        1.0e5, 1.0, num=5, dtype=np.float64
        )
    rho: np.ndarray = np.linspace(
        1.0e2, 1.0e-3, num=5, dtype=np.float64
        )
    T: np.ndarray = np.linspace(
        1.0e8, 1.0e3, num=5, dtype=np.float64
        )
    # -- additional and derived types
    G: np.float64 = np.array(
        [6.674299999999999e-8], dtype=np.float64
        )[0]
    # times R_star = 1e12 for dimensionalization
    xi_r_dim: np.ndarray = np.array(
        [(0., 0.), (1.5e12, 0.), (1.e12, 0.),
         (0.5e12, 0.), (1.25e12, 0.)],
        dtype=void_type
        )
    xi_h_dim: np.ndarray = np.array(
        [(0., 0.), (1.5e12, 0.), (1.e12, 0.),
         (0.5e12, 0.), (1.25e12, 0.)],
        dtype=void_type
        )
    # GM/R = 1.001145e15 for dimensionalization
    eul_phi_dim: np.ndarray = \
        1.001145e15 * template_complex_arr
    # GM/(R*R) = 1.001145e3 for dimensionalization
    deul_phi_dim: np.ndarray = \
        1001.145 * template_complex_arr
    # times L_star = 2e37 for dimensionalization
    lag_L_dim: np.ndarray = 2.0e37 * template_complex_arr
    # no dim
    eul_P_dim: np.ndarray = template_complex_arr
    eul_rho_dim: np.ndarray = template_complex_arr
    eul_T_dim: np.ndarray = template_complex_arr
    lag_P_dim: np.ndarray = template_complex_arr
    lag_rho_dim: np.ndarray = template_complex_arr
    lag_T_dim: np.ndarray = template_complex_arr
    # SQRT(GM/(R*R*R)) = 3.16408755e-5 for dimensionalization
    Omega_rot_dim: np.ndarray = \
        3.16408755e-5 * template_double_arr
    
    # class method used to obtain specific format values
    @classmethod
    def _get_spec_format(cls, my_key: str, mode_nr: int) -> typing.Any:
        """Gets a value from a key in a specific format.

        Parameters
        ----------
        my_key : str
            The key with which the value is retrieved.
        mode_nr : int
            The number of the mode for which you are retrieving outputs.

        Returns
        -------
        typing.Any
            The value in a specific, required format.
        """
        if ('freq' in my_key) or ('omega' == my_key):
            return getattr(cls, my_key)[mode_nr - 1].reshape(-1, 1)
        else:
            return getattr(cls, my_key)[mode_nr - 1]
        
    
    # class method used to obtain all attributes of this class
    # in a dictionary format
    @classmethod
    def get_dict(cls, mode_nr: int, nad: bool) -> dict[str, typing.Any]:
        """Retrieves all attributes of this class in a dictionary format.
        
        Parameters
        ----------
        mode_nr : int
            The number of the mode for which you are retrieving outputs.
        nad : bool
            True if you are retrieving nonadiabatic GYRE dictionaries. False if you are retrieving adiabatic GYRE dictionaries.
            
        Returns
        -------
        dict[str, typing.Any]
            The requested GYRE dictionary.
        """
        # store a temporary list of keys for which indexing must be
        # performed
        index_list = ['n', 'n_g', 'n_pg', 'omega']
        # retrieve the requested dictionary
        if nad:
            return {
                'lambda' if ('my_lambda' == a) else 
                'freq' if ('freq_complex' == a) else a: 
                    cls._get_spec_format(a, mode_nr)
                    if (a in index_list) or (a == 'freq_complex') 
                    else getattr(cls, a).reshape(-1,1)[mode_nr -1] if ('my_lambda' == a)
                    else getattr(cls, a)
                for a in vars(cls) if ('__' not in a)
                and ('get_dict' != a) and ('freq' != a)
                }            
        else:
            return {
                'lambda' if ('my_lambda' == a) else a: 
                    cls._get_spec_format(a, mode_nr)
                    if (a in index_list) or ('freq' == a)
                    else getattr(cls, a).reshape(-1,1)[mode_nr -1] if ('my_lambda' == a)
                    else getattr(cls, a)
                for a in vars(cls) if ('__' not in a) 
                and ('get_dict' != a) and ('freq_complex' != a)
                }
