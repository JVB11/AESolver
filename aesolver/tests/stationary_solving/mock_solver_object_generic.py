'''Python module used to create a mock object that mimics the quadratic coupling coefficient solver object.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
'''
# perform typing with strings
from __future__ import annotations
# import statements
import types
import typing
import numpy as np
# import typing information
if typing.TYPE_CHECKING:
    from aesolver.coupling_coefficients import QCCR


# mock class
class MockSolverGeneric:
    """Mimics the solver object used in aesolver, to test the generic statsolver functionalities.
    
    Parameters
    ----------
    nr_modes : int, optional
        The number of modes involved in the coupling; by default 3.
    """

    # attribute type declarations
    _coupler: QCCR | types.SimpleNamespace
    _lag_var_surf_lum: np.ndarray
    _xir_terms_norm: np.ndarray
    _tk_disc_integrals: np.ndarray
    _ek_disc_integrals: np.ndarray
    
    def __init__(self, nr_modes: int=3) -> None:
        # create template arrays
        _template_arr = np.ones(
            (nr_modes,), dtype=np.float64
            )
        _template_arr_complex = \
            _template_arr.copy().astype(np.complex128)
        # use these arrays to initialize the array
        # attributes
        self._lag_var_surf_lum = \
            _template_arr_complex.copy()
        self._xir_terms_norm = \
            _template_arr_complex.copy()
        self._tk_disc_integrals = \
            _template_arr.copy()
        self._ek_disc_integrals = \
            _template_arr.copy()
        # create empty coupler object (no content needed
        # for generic test)
        self._coupler = types.SimpleNamespace()


# store the expected attribute values
class ExpectedAttributes:
    """Stores the expected attribute values"""
    
    _a1: types.NoneType = None
    _a2: types.NoneType = None
    _a3: types.NoneType = None
    _f1: types.NoneType = None
    _f2: types.NoneType = None
    _f3: types.NoneType = None
    _nr_modes: int = 3
    _mode_range: range = range(1, 4)
    _coupler: QCCR | types.SimpleNamespace = \
        types.SimpleNamespace()
    _surf_lums: np.ndarray = np.ones(
        (3,), dtype=np.complex128
        )
    _surf_xirs: np.ndarray = np.ones(
        (3,), dtype=np.complex128
        )
    _tk: np.ndarray = np.ones(
        (3,), dtype=np.float64
        )
    _ek: np.ndarray = np.ones(
        (3,), dtype=np.float64
        )
    _disc_multi_factors: np.ndarray = np.zeros(
        (3,), dtype=np.float64
        )
    _a_arr: np.ndarray = np.array(
        [None, None, None], dtype=object
        )
    _f_arr: np.ndarray = np.array(
        [None, None, None], dtype=object
        )
    
    @classmethod
    def get_dict(cls) -> dict[str, typing.Any]:
        """Retrieves the expected attributes in a dictionary form.

        Returns
        -------
        dict[str, typing.Any]
            Contains the expected attributes.
        """
        return {
            _k: _v for _k, _v in cls.__dict__.items() 
            }
