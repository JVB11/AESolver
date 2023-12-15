"""Python enumeration module used to convert z-derivatives into r-derivatives of polytropic structure model quantities, and generate the output dictionary of the additional structure calculations.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
from enum import Enum
from operator import attrgetter

# type checking imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # type check imports
    from collections.abc import Callable

    # import the object used for the functions
    from ..poly_struct_comps import (
        PolytropicAdditionalStructureInformation as PaSi,
    )

    # import numpy typing
    import numpy as np
    import numpy.typing as npt


# enumeration class used to generate the output dictionary
class PolytropicStructureOutput(Enum):
    """Enumeration class that contains the necessary getter functions used to store attributes."""

    # define the enumeration members
    DRHO = attrgetter('_drho')
    DRHO2 = attrgetter('_drho2')
    DRHO3 = attrgetter('_drho3')
    DP = attrgetter('_dp')
    DP2 = attrgetter('_dp2')
    DP3 = attrgetter('_dp3')
    DGRAVPOT = attrgetter('_dgrav_pot')
    DGRAVPOT2 = attrgetter('_dgrav_pot2')
    DGRAVPOT3 = attrgetter('_dgrav_pot3')
    P = attrgetter('_p')
    RHO = attrgetter('_rho')
    R = attrgetter('_r')
    V2 = attrgetter('_v_2')
    V = attrgetter('_v')
    U = attrgetter('_u')
    C1 = attrgetter('_c1')
    AS = attrgetter('_as')
    MR = attrgetter('_mr')
    SURFGRAV = attrgetter('_surf_grav')
    BRUNT_N2 = attrgetter('_brunt_squared')
    P0 = attrgetter('_p0')
    RHO0 = attrgetter('_rho0')

    # retrieve the output dictionary
    @classmethod
    def get_output_creation_info_dict(
        cls
    ) -> 'dict[str, Callable[[PaSi], float | npt.NDArray[np.float64]]]':
        """Retrieves the enumeration members and their values and stores them in a info dictionary.

        Returns
        -------
        dict[str, Callable[[PaSi], float | npt.NDArray[np.float64]]]
            Contains information used to generate the output dictionary.
        """
        return {_k.lower(): _ev.value for _k, _ev in cls.__members__.items()}


def get_grav_pot_factor(x: 'PaSi') -> float:
    return x._p0 / x._rho0


# enumeration class used to convert derivatives
class PolytropicDerivativeConversion(Enum):
    """Python enumeration class that contains information on the conversion of z-derivatives into r-derivatives."""

    # define the enumeration members
    DRHO_DZ = ('_drho_dz', '_drho', 1, attrgetter('_rho0'))
    DRHO2_DZ2 = ('_drho2_dz2', '_drho2', 2, attrgetter('_rho0'))
    DRHO3_DZ3 = ('_drho3_dz3', '_drho3', 3, attrgetter('_rho0'))
    DP_DZ = ('_dp_dz', '_dp', 1, attrgetter('_p0'))
    DP2_DZ2 = ('_dp2_dz2', '_dp2', 2, attrgetter('_p0'))
    DP3_DZ3 = ('_dp3_dz3', '_dp3', 3, attrgetter('_p0'))
    DGRAV_POT_DZ = ('_dgrav_pot_dz', '_dgrav_pot', 1, get_grav_pot_factor)
    DGRAV_POT2_DZ2 = (
        '_dgrav_pot2_dz2',
        '_dgrav_pot2',
        2,
        get_grav_pot_factor,
    )
    DGRAV_POT3_DZ3 = (
        '_dgrav_pot3_dz3',
        '_dgrav_pot3',
        3,
        get_grav_pot_factor,
    )

    # method that retrieves the enumeration members in a list that contains information for conversion
    @classmethod
    def get_der_conversion_information(
        cls
    ) -> 'list[tuple[str, str, int, Callable[[PaSi], float]]]':
        """Method that gathers the necessary information for conversion of z-derivatives into their r counterparts.

        Return
        ------
        list[tuple[str, str, int, Callable[[PolytropicAdditionalStructureInformation], float]]]
            Contains the information tuples.
        """
        return list(map(attrgetter('value'), cls.__members__.values()))
