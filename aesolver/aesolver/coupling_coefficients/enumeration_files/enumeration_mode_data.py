"""Python file that contains enumeration classes that depict the information to be used for computing coupling coefficients.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import re
import numpy as np
from enum import Enum
from operator import attrgetter
from functools import partial

# import intra-package module: contains info on how to access GYRE detail file data
from ...model_handling.enumeration_files import (
    GYREDetailFiles,
    PolytropeOscillationModelFiles,
)

# import custom function to convert GYRE data to numpy array
from carr_conv import re_im

# type checking imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # coupling object
    from ..quadratic_coupling_coefficient_rotating import (
        QuadraticCouplingCoefficientRotating as QCCR,
    )

    # type hint check types
    from typing import Literal

    # numpy typing
    import numpy.typing as npt


# dimensionalization function library
def rim_dim(
    val: 'np.ndarray[tuple[Literal[2], ...], np.dtype[np.float64]]',
    dim: 'float | npt.NDArray[np.float64]',
) -> 'npt.NDArray[np.complex128]':
    return re_im(val, dim)


def reg_dim(
    val: 'npt.NDArray[np.float64]', dim: 'float | npt.NDArray[np.float64]'
) -> 'npt.NDArray[np.float64]':
    return val * dim


def eul_dim(
    val: 'np.ndarray[tuple[Literal[2], ...], np.dtype[np.float64]]',
    dim: 'npt.NDArray[np.float64]',
) -> 'npt.NDArray[np.complex128]':
    return re_im(val, dim[0] * dim[1] / dim[2])


def deul_dim(
    val: 'np.ndarray[tuple[Literal[2], ...], np.dtype[np.float64]]',
    dim: 'npt.NDArray[np.float64]',
) -> 'npt.NDArray[np.complex128]':
    return re_im(val, dim[0] * dim[1] / (dim[2] ** 2.0))


def e_dim(
    val: 'npt.NDArray[np.float64]', dim: 'npt.NDArray[np.float64]'
) -> 'npt.NDArray[np.float64]':
    return val * dim[0] * (dim[1] ** 2.0)


def h_dim(
    val: 'npt.NDArray[np.float64]', dim: 'npt.NDArray[np.float64]'
) -> 'npt.NDArray[np.float64]':
    return val * dim[0] * (dim[1] ** 2.0) / dim[2]


def om_dim(
    val: 'npt.NDArray[np.float64]', dim: 'npt.NDArray[np.float64]'
) -> 'npt.NDArray[np.float64]':
    return val * np.sqrt((dim[0] * dim[1]) / (dim[2] ** 3.0))


# class that holds data to be used for computing mode coupling coefficients
class ModeData(Enum):
    """Python enumeration class that contains methods to generate and identify the necessary data to compute the mode coupling coefficients."""

    # store the attributes that are common among modes, and should only be loaded once
    COMMON_ATTRIBUTES_MODES = GYREDetailFiles.retrieve_mode_independent_names(
        upper=False
    )
    COMMON_ATTRIBUTES_POLYTROPE_MODES = (
        PolytropeOscillationModelFiles.retrieve_mode_independent_names()
    )

    # store pathfinder regex
    PATH_REGEX = re.compile(r'(_mode_\d)')

    # store the functional links to dimensioning attributes
    X_DIM_ATTR = attrgetter('_R_star')
    XI_R_REF_DIM_ATTR = attrgetter('_R_star')
    XI_H_REF_DIM_ATTR = attrgetter('_R_star')
    XI_R_DIM_ATTR = attrgetter('_R_star')
    XI_H_DIM_ATTR = attrgetter('_R_star')
    EUL_PHI_REF_DIM_ATTR = attrgetter('_G', '_M_star', '_R_star')
    DEUL_PHI_REF_DIM_ATTR = attrgetter('_G', '_M_star', '_R_star')
    EUL_PHI_DIM_ATTR = attrgetter('_G', '_M_star', '_R_star')
    DEUL_PHI_DIM_ATTR = attrgetter('_G', '_M_star', '_R_star')
    LAG_S_REF_DIM_ATTR = attrgetter('_R_star')
    LAG_S_DIM_ATTR = attrgetter('_R_star')
    LAG_L_REF_DIM_ATTR = attrgetter('_L_star')
    LAG_L_DIM_ATTR = attrgetter('_L_star')
    EUL_P_DIM_ATTR = attrgetter('_P')
    EUL_RHO_DIM_ATTR = attrgetter('_rho')
    EUL_T_DIM_ATTR = attrgetter('_T')
    LAG_P_DIM_ATTR = attrgetter('_P')
    LAG_RHO_DIM_ATTR = attrgetter('_rho')
    LAG_T_DIM_ATTR = attrgetter('_T')
    E_DIM_ATTR = attrgetter('_M_star', '_R_star')
    E_P_DIM_ATTR = attrgetter('_M_star', '_R_star')
    E_G_DIM_ATTR = attrgetter('_M_star', '_R_star')
    H_DIM_ATTR = attrgetter('_G', '_M_star', '_R_star')
    W_DIM_ATTR = attrgetter('_G', '_M_star', '_R_star')
    W_EPS_DIM_ATTR = attrgetter('_G', '_M_star', '_R_star')
    TAU_SS_DIM_ATTR = attrgetter('_G', '_M_star', '_R_star')
    TAU_TR_DIM_ATTR = attrgetter('_G', '_M_star', '_R_star')
    DE_DX_DIM_ATTR = attrgetter('_M_star', '_R_star')
    DW_DX_DIM_ATTR = attrgetter('_G', '_M_star', '_R_star')
    DW_EPS_DX_DIM_ATTR = attrgetter('_G', '_M_star', '_R_star')
    DTAU_SS_DX_DIM_ATTR = attrgetter('_G', '_M_star', '_R_star')
    DTAU_TR_DX_DIM_ATTR = attrgetter('_G', '_M_star', '_R_star')
    DELTA_P_DIM_ATTR = attrgetter('_G', '_M_star', '_R_star')
    DELTA_G_DIM_ATTR = attrgetter('_G', '_M_star', '_R_star')
    OMEGA_ROT_DIM_ATTR = attrgetter('_G', '_M_star', '_R_star')

    # store the dimensioning actions
    X_DIMMER = partial(reg_dim)
    XI_R_REF_DIMMER = partial(rim_dim)
    XI_H_REF_DIMMER = partial(rim_dim)
    XI_R_DIMMER = partial(rim_dim)
    XI_H_DIMMER = partial(rim_dim)
    EUL_PHI_REF_DIMMER = partial(eul_dim)
    DEUL_PHI_REF_DIMMER = partial(deul_dim)
    EUL_PHI_DIMMER = partial(eul_dim)
    DEUL_PHI_DIMMER = partial(deul_dim)
    LAG_S_REF_DIMMER = partial(rim_dim)
    LAG_S_DIMMER = partial(rim_dim)
    LAG_L_REF_DIMMER = partial(rim_dim)
    LAG_L_DIMMER = partial(rim_dim)
    EUL_P_DIMMER = partial(rim_dim)
    EUL_RHO_DIMMER = partial(rim_dim)
    EUL_T_DIMMER = partial(rim_dim)
    LAG_P_DIMMER = partial(rim_dim)
    LAG_RHO_DIMMER = partial(rim_dim)
    LAG_T_DIMMER = partial(rim_dim)
    E_DIMMER = partial(e_dim)
    E_P_DIMMER = partial(e_dim)
    E_G_DIMMER = partial(e_dim)
    H_DIMMER = partial(h_dim)
    W_DIMMER = partial(h_dim)
    W_EPS_DIMMER = partial(h_dim)
    TAU_SS_DIMMER = partial(h_dim)
    TAU_TR_DIMMER = partial(h_dim)
    DE_DX_DIMMER = partial(e_dim)
    DW_DX_DIMMER = partial(h_dim)
    DW_EPS_DX_DIMMER = partial(h_dim)
    DTAU_SS_DX_DIMMER = partial(h_dim)
    DTAU_TR_DX_DIMMER = partial(h_dim)
    DELTA_P_DIMMER = partial(om_dim)
    DELTA_G_DIMMER = partial(om_dim)
    OMEGA_ROT_DIMMER = partial(om_dim)

    # perform dimensioning
    @classmethod
    def dimension(cls, my_object: 'QCCR') -> None:
        """Retrieve the dimensioning information + perform dimensioning and set dimensioned attributes.

        Parameters
        ----------
        my_object : QCCR
            The object to which dimensioned data shall be set.
        """
        # obtain the names of the enumeration attributes that need to be dimensioned
        _names_enumeration_dimensioned = [
            _n for _n in cls.__members__ if 'DIM_ATTR' in _n
        ]
        # obtain the names of the actual attributes in the object
        _names_attrs = [
            f"_{_n.split('_DIM_ATTR')[0]}"
            for _n in _names_enumeration_dimensioned
        ]
        # retrieve the dimensioning attributes of the object, and store their dimensioned version in a dictionary
        _dim_dict = {}
        for _attr_o in vars(my_object):
            if (
                ('__' not in _attr_o)
                and (
                    (my_name := cls.PATH_REGEX.value.sub('', _attr_o).upper())
                    in _names_attrs
                )
                and ((undim := getattr(my_object, _attr_o)) is not None)
            ):
                # retrieve the dimensioning attribute(s)
                _dim_attr_getter = cls.__getitem__(
                    f'{my_name[1:]}_DIM_ATTR'
                ).value
                if TYPE_CHECKING:
                    assert isinstance(_dim_attr_getter, attrgetter)
                _dim_attr = _dim_attr_getter(my_object)
                # retrieve the dimension function
                _dim_function = getattr(cls, f'{my_name[1:]}_DIMMER').value
                # store the dimensioned attributes in the dictionary
                _dim_dict[f'{_attr_o}_dim'] = _dim_function(
                    val=undim, dim=_dim_attr
                )
        # use that dictionary to set the dimensioned attributes
        for _attr_name, _attr_value in _dim_dict.items():
            setattr(my_object, _attr_name, _attr_value)

    # retrieve the common attributes among modes
    @classmethod
    def get_common_attrs(cls, polytrope=False) -> list[str]:
        """Utility method used to retrieve the common attributes among the modes.

        Parameters
        ----------
        polytrope : bool, optional
            If True, return the common attributes for polytropes. If False, return those for MESA+GYRE models; by default False.

        Returns
        -------
        list[str]
            Contains names of common attributes.
        """
        if polytrope:
            return cls.COMMON_ATTRIBUTES_POLYTROPE_MODES.value
        else:
            return cls.COMMON_ATTRIBUTES_MODES.value
