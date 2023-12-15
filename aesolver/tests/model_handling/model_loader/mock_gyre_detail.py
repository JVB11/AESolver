'''Python module containing functions that create a mock GYRE detail file.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
'''
# import statements
import typing
import h5py as h5
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
class GYREDetailExpectedValuesAfterRead:
    """Holds the expected values after reading for GYRE Detail files."""
    
    # attribute type and value declarations
    # -- original input
    n: np.int32 = np.array([5], np.int32)[0]
    omega: np.void = np.array(
        [0.68], dtype=np.complex128
        )[0]
    x: np.ndarray = template_double_arr
    y_1: np.ndarray = template_complex_arr
    y_2: np.ndarray = template_complex_arr
    y_3: np.ndarray = template_complex_arr
    y_4: np.ndarray = template_complex_arr
    freq: np.void = np.array(
        [0.75], dtype=np.complex128
        )[0]
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
    n_g: np.int32 = \
        np.array([23, 24, 25], np.int32)
    n_pg: np.int32 = \
        np.array([-23, -24, -25], np.int32)
    # TODO: also convert this set of void arrays to complex arrays???
    xi_r: np.ndarray = template_void_arr
    xi_h: np.ndarray = template_void_arr
    eul_phi: np.ndarray = template_void_arr
    deul_phi: np.ndarray = template_void_arr
    lag_L: np.ndarray = template_void_arr
    eul_P: np.ndarray = template_void_arr
    eul_rho: np.ndarray = template_void_arr
    eul_T: np.ndarray = template_void_arr
    lag_P: np.ndarray = template_void_arr
    lag_rho: np.ndarray = template_void_arr
    lag_T: np.ndarray = template_void_arr
    my_lambda: np.ndarray = template_complex_arr
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
    M_r: np.ndarray = template_double_arr
    P: np.ndarray = template_double_arr
    rho: np.ndarray = template_double_arr
    T: np.ndarray = template_double_arr  
    # -- additional and derived types
    G: np.float64 = np.array(
        [6.674299999999999e-8], dtype=np.float64
        )[0]
    # times R_star = 1e12 for dimensionalization
    xi_r_dim: np.ndarray = 1.e12 * template_complex_arr
    xi_h_dim: np.ndarray = 1.e12 * template_complex_arr
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
        3.16408755e-5 * template_complex_arr
    
    # class method used to obtain all attributes of this class
    # in a dictionary format
    @classmethod
    def get_dict(cls, mode_nr: int) -> dict[str, typing.Any]:
        """Retrieves all attributes of this class in a dictionary format.
        
        Parameters
        ----------
        mode_nr : int
            The number of the mode for which you are retrieving outputs.
            
        Returns
        -------
        dict[str, typing.Any]
            Contains all attributes from this class.
        """
        # retrieve the dictionary
        return {
            'lambda' if ('my_lambda' == a) else a: 
                getattr(cls, a)[mode_nr - 1] 
                if (a == 'n_g') or (a == 'n_pg')
                else getattr(cls, a)
            for a in vars(cls) if ('__' not in a) and ('get_dict' != a)
        }


# create a mock GYRE detail/mode file
def create_mock_gyre_hdf5_mode_file(my_path: str,
                                    mode_n: int=25) -> None:
    """Generates GYRE mode/detail file in HDF5 format with mock data.
    
    Parameters
    ----------
    my_path : pathlib.Path or str
        Path to the location in which the file with mock data is going to be stored.
    mode_n : int, optional
        Radial order of the mode; by default 25.
    """
    # create the (temporary) HDF5 mock file
    with h5.File(my_path, 'w') as my_hdf5:
        # create the attributes for the GYRE detail file
        my_hdf5.attrs['n'] = np.array([5], np.int32)[0]
        my_hdf5.attrs['m'] = np.array([2], np.int32)[0]
        my_hdf5.attrs['l'] = np.array([2], np.int32)[0]
        my_hdf5.attrs['n_p'] = np.array([0], np.int32)[0]
        my_hdf5.attrs['n_g'] = np.array([mode_n], np.int32)[0]
        my_hdf5.attrs['n_pg'] = np.array([-mode_n], np.int32)[0]
        my_hdf5.attrs['l_i'] = np.array(
            [(1.8, 0.0)], dtype=void_type
            )[0]
        my_hdf5.attrs['L_star'] = np.array(
            [2.0e37], dtype=np.float64
            )[0]
        my_hdf5.attrs['M_star'] = np.array(
            [1.5e34], dtype=np.float64
            )[0]
        my_hdf5.attrs['R_star'] = np.array(
            [1.0e12], dtype=np.float64
            )[0]
        my_hdf5.attrs['freq'] = np.array(
            [(0.75, 0.0)], dtype=void_type
            )[0]
        my_hdf5.attrs['freq_frame'] = np.array(
            [b'INERTIAL']
            )[0]
        my_hdf5.attrs['freq_units'] = np.array(
            [b'CYC_PER_DAY']
            )[0]      
        my_hdf5.attrs['label'] = np.array(
            [b'']
            )[0]
        my_hdf5.attrs['freq_units'] = np.array(
            [b'CYC_PER_DAY']
            )[0]
        my_hdf5.attrs['omega'] = np.array(
            [(0.68, 0.0)], dtype=void_type
            )[0]
        # create the data arrays for the GYRE detail file
        my_hdf5.create_dataset(
            name="As", data=template_double_arr.copy()
            )
        my_hdf5.create_dataset(
            name="Gamma_1", data=template_double_arr.copy()
            )
        my_hdf5.create_dataset(
            name="M_r", data=template_double_arr.copy()
            )
        my_hdf5.create_dataset(
            name="Omega_rot", data=template_double_arr.copy()
            )
        my_hdf5.create_dataset(
            name="P", data=template_double_arr.copy()
            )
        my_hdf5.create_dataset(
            name="T", data=template_double_arr.copy()
            )
        my_hdf5.create_dataset(
            name="U", data=template_double_arr.copy()
            )
        my_hdf5.create_dataset(
            name="V_2", data=template_double_arr.copy()
            )
        my_hdf5.create_dataset(
            name="c_1", data=template_double_arr.copy()
            )
        my_hdf5.create_dataset(
            name="dnabla_ad", data=template_double_arr.copy()
            )
        my_hdf5.create_dataset(
            name="nabla", data=template_double_arr.copy()
            )
        my_hdf5.create_dataset(
            name="nabla_ad", data=template_double_arr.copy()
            )
        my_hdf5.create_dataset(
            name="rho", data=template_double_arr.copy()
            )
        my_hdf5.create_dataset(
            name="x", data=template_double_arr.copy()
            )
        my_hdf5.create_dataset(
            name="deul_phi", data=template_void_arr.copy()
            )
        my_hdf5.create_dataset(
            name="eul_P", data=template_void_arr.copy()
            )
        my_hdf5.create_dataset(
            name="eul_T", data=template_void_arr.copy()
            )
        my_hdf5.create_dataset(
            name="eul_phi", data=template_void_arr.copy()
            )
        my_hdf5.create_dataset(
            name="eul_rho", data=template_void_arr.copy()
            )
        my_hdf5.create_dataset(
            name="lag_L", data=template_void_arr.copy()
            )
        my_hdf5.create_dataset(
            name="lag_P", data=template_void_arr.copy()
            )
        my_hdf5.create_dataset(
            name="lag_T", data=template_void_arr.copy()
            )
        my_hdf5.create_dataset(
            name="lag_rho", data=template_void_arr.copy()
            )
        my_hdf5.create_dataset(
            name="lambda", data=template_void_arr.copy()
            )
        my_hdf5.create_dataset(
            name="xi_h", data=template_void_arr.copy()
            )
        my_hdf5.create_dataset(
            name="xi_r", data=template_void_arr.copy()
            )
        my_hdf5.create_dataset(
            name="y_1", data=template_void_arr.copy()
            )
        my_hdf5.create_dataset(
            name="y_2", data=template_void_arr.copy()
            )
        my_hdf5.create_dataset(
            name="y_3", data=template_void_arr.copy()
            )
        my_hdf5.create_dataset(
            name="y_4", data=template_void_arr.copy()
            )
