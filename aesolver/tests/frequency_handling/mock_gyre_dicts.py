'''Python module containing classes that hold gyre input information in dictionaries.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
'''
# import statements
import typing
import numpy as np


# define the Mock Gyre Input object for adiabatic GYRE files
class MockGYREAdiabaticInputObject:
    """Creates a mocking class that holds the information necessary to create the input  dictionaries for the frequency handler object."""
    
    # attribute declarations
    freq_units: bytes = b'HZ'
    Omega_rot_dim: np.ndarray = \
        0.5 * np.ones((5,), dtype=np.float64)
    G: np.float64 = \
        np.array([1.0], dtype=np.float64)
    M_star: np.float64 = \
        np.array([1.0], dtype=np.float64)
    R_star: np.float64 = \
        np.array([1.0], dtype=np.float64)
    freq: np.ndarray = \
        np.array([2.0, 3.0, 4.0], dtype=np.complex128)
    m: np.ndarray = \
        np.array([2, 2, 2], dtype=np.int32)
    freq_frame: np.bytes_ = \
        np.array([b'INERTIAL'], dtype=np.bytes_)
    
    # define a generic dictionary for each mode
    @classmethod
    def _get_generic_dict(cls, nr_mode: int=1) -> dict[str, typing.Any]:
        """Retrieve the information useful for a  GYRE input dictionary.

        Parameters
        ----------
        nr_mode : int, optional
            The number of the mode for which you are constructing the GYRE input dictionary; by default 1.
            
        Returns
        -------
        dict
            The constructed GYRE input dictionary for the mode with number 'nr_mode'.
        """
        # construct the dictionary and return it
        return {
            'freq_units': cls.freq_units,
            'Omega_rot_dim': cls.Omega_rot_dim,
            'G': cls.G, 'M_star': cls.M_star,
            'R_star': cls.R_star,
            'm': cls.m[nr_mode - 1],
            # reshape necessary to make code work at this time --> TODO: fix this!
            'freq': cls.freq[nr_mode - 1].reshape(-1,1),  
            'freq_frame': cls.freq_frame
            }
        
    # retrieves the list of GYRE dictionaries for the
    # freq_handler object
    @classmethod
    def get_gyre_dicts(cls) -> list[dict[str, typing.Any]]:
        """Constructs the list of GYRE input dictionaries used for the freq_handler object.

        Returns
        -------
        list[dict]
            List of GYRE input dictionaries.
        """
        return [
            cls._get_generic_dict(nr_mode=_i)
            for _i in range(1, 4)
            ]


# define the Mock Gyre Input object for non-adiabatic GYRE files
class MockGYRENonAdiabaticInputObject:
    """Creates a mocking class that holds the information necessary to create the input  dictionaries for the frequency handler object."""
    
    # attribute declarations
    freq_units: bytes = b'HZ'
    freq: np.ndarray = np.array(
        [2.0+0.1j, 3.0-0.25j, 4.0-0.5j], dtype=np.complex128
        )
    m: np.ndarray = np.array([2, 2, 2], dtype=np.int32)
    freq_frame: np.bytes_ = np.array(
        [b'INERTIAL'], dtype=np.bytes_
        )
    
    # define a generic dictionary for each mode
    @classmethod
    def _get_generic_dict(cls, nr_mode: int=1) -> dict[str, typing.Any]:
        """Retrieve the information useful for a GYRE input dictionary.

        Parameters
        ----------
        nr_mode : int, optional
            The number of the mode for which you are constructing the GYRE input dictionary; by default 1.
            
        Returns
        -------
        dict
            The constructed GYRE input dictionary for the mode with number 'nr_mode'.
        """
        # construct the dictionary and return it
        return {
            'm': cls.m[nr_mode - 1],
            # reshape necessary to make code work at this time --> TODO: fix this!
            'freq': cls.freq[nr_mode - 1].reshape(-1,1),
            'freq_frame': cls.freq_frame,
            'freq_units': cls.freq_units
            }
        
    # retrieves the list of GYRE dictionaries for the
    # freq_handler object
    @classmethod
    def get_gyre_dicts(cls) -> list[dict[str, typing.Any]]:
        """Constructs the list of GYRE input dictionaries used for the freq_handler object.

        Returns
        -------
        list[dict]
            List of GYRE input dictionaries.
        """
        return [
            cls._get_generic_dict(nr_mode=_i)
            for _i in range(1, 4)
            ]
