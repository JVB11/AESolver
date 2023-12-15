"""Python module containing generic/superclass used for computation of the stationary solutions of the AEs.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# ensure type annotations are not evaluated
# but instead are handled as strings
from __future__ import annotations

# import statements
import numpy as np
import logging

# typing imports
from typing import TYPE_CHECKING

# import necessary modules for type checking
if TYPE_CHECKING:
    # import intra-package coupling-coefficient-computing module
    from ...coupling_coefficients import QCCR

    # import solver objects (only quadratic rotating AE solver for now)
    # -- quadratic rotating AE solver
    from ...solver import QuadRotAESolver

    # import numpy typing
    import numpy.typing as npt

# set up logger
logger = logging.getLogger(__name__)


# generic/superclass used for computation of stationary solutions
class StatSolve:
    """Generic/superclass used for computation of the stationary solutions of the AEs for the different mode combinations."""

    # attribute type declarations
    _coupler: 'QCCR | None'
    _qf_products: 'npt.NDArray[np.float64]'
    _nr_modes: int
    _mode_range: range
    _surf_lums: 'npt.NDArray[np.float64]'
    _surf_xirs: 'npt.NDArray[np.float64]'

    def __init__(
        self, solver_object: 'QuadRotAESolver', nr_modes: int = 3
    ) -> None:
        # use the solver object to obtain the relevant attributes
        # - retrieve the mode coupling object
        self._coupler = solver_object._coupler
        # store my mode range
        self._nr_modes = nr_modes
        self._mode_range = range(1, nr_modes + 1)
        # initialize attributes that will hold the amplitudes
        self._initialize_mode_attributes('a')
        # initialize attributes that will hold the luminosity fluctuations
        self._initialize_mode_attributes('f')
        # store attributes used to compute (Delta L_alpha(R)/L(R))
        self._surf_lums = solver_object._lag_var_surf_lum
        self._surf_xirs = solver_object._xir_terms_norm
        # store the computed disc integral factors
        self._tk = solver_object._tk_disc_integrals
        self._ek = solver_object._ek_disc_integrals

    def _compute_disc_integral_multiplication_factors(self) -> None:
        """Compute the disc integral multiplication factors."""
        # compute and store the disc multiplication factors, that is, the modulus of the mode-dependent factor
        self._disc_multi_factors = np.abs(
            self._tk * self._surf_lums - self._ek * self._surf_xirs
        )

    def _create_array_from_attrs(
        self, attrname: str
    ) -> 'npt.NDArray[np.float64]':
        """Creates array from attributes stored in instance.

        Parameters
        ----------
        attrname : str
            Generic name of the attribute(s).

        Returns
        -------
        npt.NDArray[np.float64]
            Numpy array containing the mode attributes.
        """
        return np.array(
            [getattr(self, f'{attrname}{_i}') for _i in self._mode_range]
        )

    def _initialize_mode_attributes(self, attr_string: str) -> None:
        """Initializes an instance attribute of the modes with name given by the input 'attr_string' to None values.

        Parameters
        ----------
        attr_string : str
            Name of the mode attributes to be initialized to None values.
        """
        for _i in self._mode_range:
            setattr(self, f'_{attr_string}{_i}', None)
