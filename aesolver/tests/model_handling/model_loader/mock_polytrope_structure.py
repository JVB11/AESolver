'''Python module containing functions that create a mock GYRE polytrope structure file.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
'''
# import statements
import typing
import h5py as h5
import numpy as np
from astropy.constants import G


# define some variables used in expected property enum
M_star = 3.0
R_star = 5.0
G_val = G.cgs.value
z_val = np.linspace(
    0.1, 2.5, num=5, dtype=np.float64
    )
x_val = z_val / z_val[-1]


# create a mock GYRE polytrope structure file
def create_mock_gyre_hdf5_polytrope_structure_file(
    my_path: str
    ) -> None:
    """Generates a GYRE polytrope structure file in HDF5 format with mock data.

    Parameters
    ----------
    my_path : str
        Path to the location in which the file with mock data is going to be stored.
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
            data=-0.25 * np.ones(
                (5,), dtype=np.float64
                )
            )
        my_hdf5.create_dataset(
            name='theta',
            data=np.linspace(
                1.25, 0.25, num=5, dtype=np.float64
                )
            )
        my_hdf5.create_dataset(
            name='z',
            data=np.linspace(
                0.1, 2.5, num=5, dtype=np.float64
                )
            )


# create a object that returns the expected values
# for each parameter
class GYREPolytropeStructureExpectedValuesAfterLoad:
    """Holds the expected values after reading the GYRE polytrope structure file."""
    
    # attribute type and value declarations
    n: np.int32 = np.array([5], dtype=np.int32)[0]
    n_poly: np.float64 = \
        np.array([3.0], dtype=np.float64)[0]
    n_r: np.int32 = np.array([1], dtype=np.int32)[0]
    Gamma_1: np.float64 = \
        np.array([5./3.], dtype=np.float64)[0]
    dtheta: np.ndarray = \
        -0.25 * np.ones((5,), dtype=np.float64)
    theta: np.ndarray = \
       np.linspace(1.25, 0.25, num=5, dtype=np.float64)
    z: np.ndarray = np.linspace(
        0.1, 2.5, num=5, dtype=np.float64
        )
    G: np.float64 = G_val
    drho: np.ndarray = np.array(
        [-0.01119058, -0.00716197, -0.00402861,
         -0.00179049, -0.00044762]
        )
    drho2: np.ndarray = np.array(
        [ 0.07043073, -0.00230206,  0.00104266, 
         0.00138999,  0.00061268]
        )
    drho3: np.ndarray = np.array(
        [-1.26948060e+00,  6.75545742e-03, 
         8.15000481e-04, -1.13858746e-03,
         -7.99846844e-04]
        )
    dp: np.ndarray = np.array(
        [-2.98757204e-10, -1.52963688e-10, -6.45315561e-11,
         -1.91204611e-11, -2.39005763e-12]
        )
    dp2: np.ndarray = np.array(
        [ 1.91017887e-09, -3.00464388e-11,  2.74569361e-11, 
         1.96236311e-11, 4.46642020e-12]
        )
    dp3: np.ndarray = np.array(
        [-3.44496932e-08,  1.67499141e-10,  8.28918657e-12,
         -2.09014181e-11, -7.98279249e-12]
        )
    dgravpot: np.ndarray = np.array(
        [8.00916e-09, 8.00916e-09, 8.00916e-09, 8.00916e-09,
         8.00916e-09]
        )
    dgravpot2: np.ndarray = np.array(
        [-4.88058187e-08,  4.57666286e-09,  5.96836442e-10,
         -2.21305737e-09, -2.95337775e-09]
        )
    dgravpot3: np.ndarray = np.array(
        [ 8.79130453e-07, -4.37234755e-09, -1.46839537e-09, 
         7.72351475e-10, 1.44665452e-09]
        )
    p: np.ndarray = np.array(
        [7.46893010e-10, 3.05927377e-10, 9.67973341e-11,
         1.91204611e-11, 1.19502882e-12]
        )
    rho: np.ndarray = np.array(
        [0.03730194, 0.01909859, 0.00805722, 0.00238732,
         0.00029842]
        )
    r: np.ndarray = np.array(
        [0.2, 1.4, 2.6, 3.8, 5. ]
        )
    v2: np.ndarray = np.array(
        [50.        ,  8.92857143,  6.41025641, 
         6.57894737, 10.        ]
        )
    v: np.ndarray = np.array(
        [ 0.08      ,  0.7       ,  1.73333333, 
         3.8       , 10.        ]
        )
    u: np.ndarray = np.array(
        [0.78125, 2.8    , 2.19375, 0.95   , 0.15625]
        )
    c1: np.ndarray = np.array(
        [0.04, 0.28, 0.52, 0.76, 1.  ]
        )
    AS: np.ndarray = np.array(
        [0.012, 0.105, 0.26 , 0.57 , 1.5  ]
        )
    mr: np.ndarray = np.array(
        [0.0048, 0.2352, 0.8112, 1.7328, 3.    ]
        )
    surfgrav: np.ndarray = np.array(
        [8.00916e-09, 8.00916e-09, 8.00916e-09, 8.00916e-09,
         8.00916e-09]
        )
    brunt_n2: np.ndarray = (np.array(
        [8.00916e-09, 8.00916e-09, 8.00916e-09, 8.00916e-09,
         8.00916e-09]
        ) * np.array(
            [0.012, 0.105, 0.26 , 0.57 , 1.5  ]
            )) / (R_star * x_val)
    p0: np.float64 = 3.0592737696333214e-10
    rho0: np.float64 = 0.01909859317102744
    
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
            'as' if (a == 'AS') else a:
                getattr(cls, a) for a in vars(cls)
            if ('__' not in a) and ('get_dict' != a)
        }
