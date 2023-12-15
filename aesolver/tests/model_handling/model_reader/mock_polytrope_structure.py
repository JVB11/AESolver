'''Python module containing functions that create a mock GYRE polytrope structure file.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
'''
# import statements
import typing
import h5py as h5
import numpy as np


# create a mock GYRE polytrope structure file
def create_mock_gyre_hdf5_polytrope_structure_file(
    my_path: str
    ) -> None:
    """Generates a GYRE polytrope structure file in HDF5 format with mock data.

    Parameters
    ----------
    my_path : str
        Path to the location in which the file with  mock data is going to be stored.
    """
    # create the (temporary) HDF5 mock file
    with h5.File(my_path, 'w') as my_hdf5:
        # create the attributes
        my_hdf5.attrs['n'] = \
            np.array([5], dtype=np.int32)[0]
        my_hdf5.attrs['n_poly'] = \
            np.array([3.0], dtype=np.float64)[0]
        my_hdf5.attrs['n_r'] = \
            np.array([1], dtype=np.int32)[0]
        my_hdf5.attrs['Gamma_1'] = \
            np.array([5./3.], dtype=np.float64)[0]
        # create the data arrays
        my_hdf5.create_dataset(
            name='dtheta',
            data=np.ones((5,), dtype=np.float64)
            )
        my_hdf5.create_dataset(
            name='theta',
            data=np.ones((5,), dtype=np.float64)
            )
        my_hdf5.create_dataset(
            name='z',
            data=np.ones((5,), dtype=np.float64)
            )


# create a object that returns the expected values
# for each parameter
class GYREPolytropeStructureExpectedValuesAfterRead:
    """Holds the expected values after reading the GYRE polytrope structure file."""
    
    # attribute type and value declarations
    n: np.int32 = np.array([5], dtype=np.int32)[0]
    n_poly: np.float64 = \
        np.array([3.0], dtype=np.float64)[0]
    n_r: np.int32 = np.array([1], dtype=np.int32)[0]
    Gamma_1: np.float64 = \
        np.array([5./3.], dtype=np.float64)[0]
    dtheta: np.ndarray = \
        np.ones((5,), dtype=np.float64)
    theta: np.ndarray = \
        np.ones((5,), dtype=np.float64)
    z: np.ndarray = np.ones((5,), dtype=np.float64)
    
    # class method used to obtain all attributes
    @classmethod
    def get_dict(cls) -> dict[str, typing.Any]:
        """Retrieves all attributes of this class in a dictionary format.

        Returns
        -------
        dict[str, typing.Any]
            Contains all attributes from this class.
        """
        return {
            a: getattr(cls, a) for a in vars(cls)
            if ('__' not in a) and ('get_dict' != a)
        }
