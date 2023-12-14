"""Contains code to generate mock input data for our test of this package.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import numpy as np

# import custom pytest utility class
from pytest_util_classes import EVs  # type: ignore


# create new subclass to generate input data enum classes
class InputDataEnums(EVs):
    """Subclass used to generate mock-up input data."""

    # get representation string
    @classmethod
    def get_custom_repr(cls):
        """Get custom representation name for the Input class, which consists of the name of the class where the 'InputData' part has been removed."""
        # remove 'InputData' from the classname to return the
        # custom representation of its name
        return f'{cls.__name__[9:]}'

    # Translate input data keys to keys used for data storage
    @classmethod
    def _get_translation_dict(cls, nr_modes=3):
        """Translates input data keys to keys used for data storage.

        Notes
        -----
        Maximum nr. of allowed modes = 3.

        Parameters
        ----------
        nr_modes : int, optional
            The number of modes considered; by default 3.

        Returns
        -------
        dict
            Translation dictionary for the keys.
        """
        # create mode-nr related dictionary
        MODE = {}
        for _i in range(1, nr_modes + 1):
            _mod_cap = f'_MODE_{_i}'
            _mod_str = f'_mode_{_i}'
            MODE[f'X_Z_1{_mod_cap}'] = f'_x_z_1{_mod_str}'
            MODE[
                f'X_Z_2_OVER_C_1_OMEGA{_mod_cap}'
            ] = f'_x_z_2_over_c_1_omega{_mod_str}'
            MODE[f'HR{_mod_cap}'] = f'_hr{_mod_str}'
            MODE[f'HT{_mod_cap}'] = f'_ht{_mod_str}'
            MODE[f'HP{_mod_cap}'] = f'_hp{_mod_str}'
        # create non-mode-nr related dictionary
        NO_MODE = {
            'G': '_G',
            'M_STAR': '_M_star',
            'R_STAR': '_R_star',
            'X': '_x',
            'RHO_NORMED': '_rho_normed',
            'NORM_RHO': '_norm_rho',
            'SURF_ROT_FREQ': '_surf_rot_freq',
            'MU_VALUES': '_mu_values',
            'COROT_MODE_OMEGAS': '_corot_mode_omegas',
            'NUMERICAL_INTEGRATION_METHOD': '_numerical_integration_method',
            'OMP_THREADS': '_omp_threads',
            'USE_PARALLEL': '_use_parallel',
        }
        # return the merged dictionary
        return MODE | NO_MODE

    # access method using dictionary
    @classmethod
    def access_all_input_data(cls, nr_modes=3):
        """Method used to access all input data stored in the enum, using a translation dictionary defined for each separate class.

        Notes
        -----
        Maximum nr. of allowed modes = 3.

        Parameters
        ----------
        nr_modes : int, optional
            The number of modes considered; by default 3.

        Returns
        -------
        dict
            Input data stored in a dictionary.
        """
        # get the enumeration dictionary
        my_enum_dict = cls.get_enumeration_dict()
        # get the translation dictionary
        my_translation_dict = cls._get_translation_dict(nr_modes=nr_modes)
        # return the translated input dictionary
        return {
            my_translation_dict[_k.upper()]: _v
            for _k, _v in my_enum_dict.items()
        }


# mock input data enum -- full of ones
class InputDataOnes(InputDataEnums):
    """Contains the mock-up input data containing unity values."""

    # define enumeration attributes
    # NEED TO CONVERT NUMPY ARRAYS AND LISTS TO HASHABLE TYPES!
    G = 1.0
    M_STAR = 1.0
    R_STAR = 1.0
    X_Z_1_MODE_1 = np.ones(5000).tobytes()
    X_Z_1_MODE_2 = np.ones(5000).tobytes()
    X_Z_1_MODE_3 = np.ones(5000).tobytes()
    X_Z_2_OVER_C_1_OMEGA_MODE_1 = np.ones(5000).tobytes()
    X_Z_2_OVER_C_1_OMEGA_MODE_2 = np.ones(5000).tobytes()
    X_Z_2_OVER_C_1_OMEGA_MODE_3 = np.ones(5000).tobytes()
    HR_MODE_1 = np.ones(5000).tobytes()
    HR_MODE_2 = np.ones(5000).tobytes()
    HR_MODE_3 = np.ones(5000).tobytes()
    HT_MODE_1 = np.ones(5000).tobytes()
    HT_MODE_2 = np.ones(5000).tobytes()
    HT_MODE_3 = np.ones(5000).tobytes()
    HP_MODE_1 = np.ones(5000).tobytes()
    HP_MODE_2 = np.ones(5000).tobytes()
    HP_MODE_3 = np.ones(5000).tobytes()
    X = np.linspace(0.0001, 1.0000, 5000).tobytes()
    RHO_NORMED = np.linspace(1.0000, 0.0001, 5000).tobytes()
    NORM_RHO = 1.0
    SURF_ROT_FREQ = 1.0
    COROT_MODE_OMEGAS = np.array([1.0, 1.0, 1.0]).tobytes()
    MU_VALUES = np.linspace(-1.0000, 1.0000, 5000).tobytes()
    NUMERICAL_INTEGRATION_METHOD = 'trapz'
    OMP_THREADS = 1
    USE_PARALLEL = False
