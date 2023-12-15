'''Python module containing functions that create a mock GYRE polytrope mode file.

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
template_double_arr = np.ones(
    (5,), dtype=np.float64
    )
template_void_arr = np.array(
    [(1.0, 0.0),(1.0, 0.0),(1.0, 0.0),(1.0, 0.0),(1.0, 0.0)],
    dtype=void_type
    )


# create a mock GYRE polytrope mode file
def create_mock_gyre_hdf5_polytrope_mode_file(
    my_path: str, input_data: dict[str, int | tuple[float]]
    ) -> None:
    """Generates a GYRE polytrope mode file in HDF5 format with mock data.

    Parameters
    ----------
    my_path : pathlib.Path or str
        Path to the location in which the file with mock data is going to be stored.
    input_data : dict[str, int | tuple[float]]
        Contains integer or float tuple input data for the individual modes.
    """
    # create the (temporary) HDF5 mock file
    with h5.File(my_path, 'w') as my_hdf5:
        # create the attributes
        my_hdf5.attrs['l'] = \
            np.array([input_data['l']],
                     dtype=np.int32)[0]
        my_hdf5.attrs['label'] = \
            np.array([b''])[0]
        my_hdf5.attrs['m'] = \
            np.array([input_data['m']],
                     dtype=np.int32)[0]
        my_hdf5.attrs['n'] = \
            np.array([5], dtype=np.int32)[0]
        my_hdf5.attrs['n_g'] = \
            np.array([input_data['n_g']],
                     dtype=np.int32)[0]
        my_hdf5.attrs['n_p'] = \
            np.array([0], dtype=np.int32)[0]
        my_hdf5.attrs['n_pg'] = \
            np.array([-input_data['n_g']],
                     dtype=np.int32)[0]
        my_hdf5.attrs['omega'] = \
            np.array([input_data['omega']],
                     dtype=void_type)[0]
        my_hdf5.attrs['freq'] = \
            np.array([input_data['omega']],
                     dtype=void_type)[0]
        my_hdf5.attrs['freq_frame'] = \
            np.array([b'INERTIAL'])[0]
        my_hdf5.attrs['freq_units'] = \
            np.array([b'NONE'])[0]
        my_hdf5.attrs['l_i'] = \
            np.array([(float(input_data['l']), 0.0)],
                     dtype=void_type)[0]
        # create the data arrays
        my_hdf5.create_dataset(
            name='Gamma_1',
            data=template_double_arr.copy()
            )
        my_hdf5.create_dataset(
            name='deul_phi',
            data=template_void_arr.copy()
            )
        my_hdf5.create_dataset(
            name='eul_phi',
            data=template_void_arr.copy()
            )
        my_hdf5.create_dataset(
            name='x',
            data=template_double_arr.copy()
            )
        my_hdf5.create_dataset(
            name='xi_h',
            data=template_void_arr.copy()
            )
        my_hdf5.create_dataset(
            name='xi_r',
            data=template_void_arr.copy()
            )
        my_hdf5.create_dataset(
            name='Omega_rot',
            data=template_double_arr.copy()
            )
        my_hdf5.create_dataset(
            name='lambda',
            data=template_void_arr.copy()
            )
        my_hdf5.create_dataset(
            name='y_1',
            data=template_void_arr.copy()
            )
        my_hdf5.create_dataset(
            name='y_2',
            data=template_void_arr.copy()
            )
        my_hdf5.create_dataset(
            name='y_3',
            data=template_void_arr.copy()
            )
        my_hdf5.create_dataset(
            name='y_4',
            data=template_void_arr.copy()
            )

# create a object that returns the expected values for each parameter
class GYREPolytropeModeExpectedValuesAfterRead:
    """Holds the expected values after reading for GYRE polytrope mode files."""
    
    # attribute type and value declarations
    l: np.int32 = np.array([2], dtype=np.int32)[0]
    l_i: void_type = np.array(
        [(2.0, 0.0)], dtype=void_type
        )[0]
    label: np.bytes_ = np.array([b''])[0]
    m: np.ndarray = \
        np.array([2, 1, 1], dtype=np.int32)
    n: np.int32 = np.array([5], dtype=np.int32)[0]
    n_g: np.ndarray = \
        np.array([23, 24, 25], dtype=np.int32)
    n_p: np.int32 = np.array([0], dtype=np.int32)[0]
    n_pg: np.ndarray = \
        np.array([-23, -24, -25], dtype=np.int32)
    omega: np.ndarray = \
        np.array([(0.75, 0.0),
                  (0.65, 0.0),
                  (0.15, 0.0)], dtype=void_type)
    freq: np.ndarray = \
        np.array([(0.75, 0.0),
                  (0.65, 0.0),
                  (0.15, 0.0)], dtype=void_type)
    freq_frame: np.bytes_ = np.array([b'INERTIAL'])[0]
    freq_units: np.bytes_ = np.array([b'NONE'])[0]
    Gamma_1: np.ndarray = template_double_arr.copy()
    deul_phi: np.ndarray = template_void_arr.copy()
    eul_phi: np.ndarray = template_void_arr.copy()
    x: np.ndarray = template_double_arr.copy()
    xi_h: np.ndarray = template_void_arr.copy()
    xi_r: np.ndarray = template_void_arr.copy()
    Omega_rot: np.ndarray = template_double_arr.copy()
    my_lambda: np.ndarray = template_void_arr.copy()
    y_1: np.ndarray = template_void_arr.copy()
    y_2: np.ndarray = template_void_arr.copy()
    y_3: np.ndarray = template_void_arr.copy()
    y_4: np.ndarray = template_void_arr.copy()
    
    # class method used to obtain all attributes of this class
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
        # mode specific list
        mode_spec_list = ['m', 'n_g', 'n_pg', 'omega', 'freq']
        # retrieve the dictionary
        return {
            'lambda' if (a == 'my_lambda') else a: 
                getattr(cls, a)[mode_nr - 1] if a in mode_spec_list
                else getattr(cls, a) for a in vars(cls)
            if ('__' not in a) and ('get_dict' != a)
            }    
    