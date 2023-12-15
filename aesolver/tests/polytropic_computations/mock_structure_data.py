'''Python module used to generate mock polytrope structure data.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
'''
# import statements
import typing
import numpy as np
from astropy.constants import G


# create a object that returns the expected values
# for each parameter
class GYREPolytropeStructureValuesAfterLoad:
    """Holds structure info values after loading the GYRE polytrope structure file."""
    
    # attribute type and value declarations
    n: np.int32 = np.array([5], dtype=np.int32)[0]
    n_poly: np.float64 = \
        np.array([3.0], dtype=np.float64)[0]
    n_r: np.int32 = np.array([1], dtype=np.int32)[0]
    Gamma_1: np.float64 = \
        np.array([5./3.], dtype=np.float64)[0]
    theta: np.ndarray = \
        np.linspace(1.25, 0.25, num=5, dtype=np.float64)
    dtheta: np.ndarray = \
        -0.25 * np.ones((5,), dtype=np.float64)
    z: np.ndarray = np.linspace(
        0.1, 2.5, num=5, dtype=np.float64
        )
    G: float = G.cgs.value
    
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