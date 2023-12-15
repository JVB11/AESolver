"""Python enumeration module containing enumeration classes that perform dimensionalization for specific input.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import logging
import numpy as np
from enum import Enum

# import custom function
from carr_conv import re_im

# typing imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


# set up logger
logger = logging.getLogger(__name__)


# enumeration superclass for dimensionalization
class DimClass(Enum):
    """Template enumeration (super)class used for dimensionalization of the terms loaded from simulation output (MESA/GYRE/...)."""

    # generic class method used to dimensionalize the simulation output
    @classmethod
    def dimensionalize(cls, my_dict: "dict[str, Any]"):
        """Generic class method used to dimensionalize the simulation output, based on the supplied dictionary.

        Parameters
        ----------
        my_dict : dict[str, Any]
            Contains the simulation output (key-word argument).
        """
        # get dummy dictionary in which dimensionalized attributes shall be stored
        _dummy_dict = {}
        # loop over the dictionary, retrieving the dimensionalization functions if necessary
        for _k, _v in my_dict.items():
            if _k not in cls.AVOID_LIST.value:  # type: ignore (defined in subclass)
                try:
                    # get dimensionalizing factor
                    _factor_dim = getattr(cls, _k.upper())(my_dict)
                    _name = f'{_k}_dim'
                except AttributeError:
                    logger.debug(f'{_k} needs no dimensionalization.')
                    _factor_dim = 1.0
                    _name = _k
                # if necessary, convert to complex array
                if _k in cls.COMPLEX_CONVERSION_LIST.value:  # type: ignore (defined in subclass)
                    _dummy_dict[_name] = re_im(_v, _factor_dim)
                else:
                    _dummy_dict[_name] = _v * _factor_dim
        # add values to the dictionary
        for _r, _f in _dummy_dict.items():
            my_dict[_r] = _f


# enumeration class for conversion of re_im values of POLYTROPE mode files
class ReImPoly(DimClass):
    """Enumeration (sub)class containing methods used to convert re_im-formatted values of POLYTROPE mode file output."""

    # add a list containing the strings of arrays that need complex array conversion
    COMPLEX_CONVERSION_LIST = [
        'xi_r',
        'xi_h',
        'eul_phi',
        'deul_phi',
        'omega',
        'y_1',
        'y_2',
        'y_3',
        'y_4',
        'y_5',
        'y_6',
        'freq',
        'l_i',
        'lambda',
    ]
    AVOID_LIST = [
        'freq_units',
        'freq_frame',
        'n',
        'j',
        'l',
        'm',
        'n_p',
        'n_g',
        'n_pg',
        'label',
    ]


# define helper functions
def get_r(d: "dict[str, Any]") -> "Any":
    """Retrieves the multiplying factor for dimensionalizing radii.

    Parameters
    ----------
    d : dict[str, Any]
        Data dictionary holding the necessary data to dimensionalize radii and related quantities.

    Returns
    -------
    Any
        The requested property stored in the data dictionary: the stellar radius and related quantities.
    """
    return d['R_star']


def get_eul_phi_fac(d: "dict[str, Any]") -> "Any":
    """Retrieves the multiplying factor for dimensionalizing the Eulerian gravitational potential perturbation and related quantities.

    Parameters
    ----------
    d : dict[str, Any]
        Data dictionary holding the necessary data to dimensionalize radii and related quantities.

    Returns
    -------
    Any
        The requested property stored in the data dictionary: the multiplying factor for the Eulerian gravitational potential perturbation and related quantities.
    """
    return d['G'] * d['M_star'] / d['R_star']


def get_deul_phi_fac(d: "dict[str, Any]") -> "Any":
    """Retrieves the multiplying factor for dimensionalizing the derivative of the Eulerian gravitational potential perturbation and related quantities.

    Parameters
    ----------
    d : dict[str, Any]
        Data dictionary holding the necessary data to dimensionalize derivatives of the Eulerian gravitational potential and related quantities.

    Returns
    -------
    Any
        The requested property stored in the data dictionary: the multiplying factor for the derivative of the Eulerian gravitational potential perturbation and related quantities.
    """
    return d['G'] * d['M_star'] / (d['R_star'] ** 2.0)


def get_P(d: "dict[str, Any]") -> "Any":
    """Retrieves the pressure.

    Parameters
    ----------
    d : dict[str, Any]
        Data dictionary holding the necessary data to dimensionalize pressures and related quantities.

    Returns
    -------
    Any
        The requested property stored in the data dictionary: the multiplying factor for the pressure, its perturbation, and related quantities.
    """
    return d['P']


def get_rho(d: "dict[str, Any]") -> "Any":
    """Retrieves the density.

    Parameters
    ----------
    d : dict[str, Any]
        Data dictionary holding the necessary data to dimensionalize densities and related quantities.

    Returns
    -------
    Any
        The requested property stored in the data dictionary: the multiplying factor for the density, its perturbation, and related quantities.
    """
    return d['rho']


def get_T(d: "dict[str, Any]") -> "Any":
    """Retrieves the temperature.

    Parameters
    ----------
    d : dict[str, Any]
        Data dictionary holding the necessary data to dimensionalize temperatures and related quantities.

    Returns
    -------
    Any
        The requested property stored in the data dictionary: the multiplying factor for the density, its perturbation, and related quantities.
    """
    return d['T']


def get_om_rot_fac(d: "dict[str, Any]") -> "Any":
    """Retrieves the conversion factor for the rotation frequency.

    Parameters
    ----------
    d : dict[str, Any]
        Data dictionary holding the necessary data to dimensionalize rotation frequencies and related quantities.

    Returns
    -------
    Any
        The requested property stored in the data dictionary: the multiplying factor for the rotation frequency, and related quantities.
    """
    return np.sqrt(d['G'] * d['M_star'] / (d['R_star'] ** 3.0))


def get_L(d: "dict[str, Any]") -> "Any":
    """Retrieves the conversion factor for the luminosity.

    Parameters
    ----------
    d : dict[str, Any]
        Data dictionary holding the necessary data to dimensionalize luminosities and related quantities.

    Returns
    -------
    Any
        The requested property stored in the data dictionary: the multiplying factor for the luminosity, and related quantities.
    """
    return d['L_star']


def get_E_fac(d: "dict[str, Any]") -> "Any":
    """Retrieves the conversion factor for the mode energy.

    Parameters
    ----------
    d : dict[str, Any]
        Data dictionary holding the necessary data to dimensionalize mode energies and related quantities.

    Returns
    -------
    Any
        The requested property stored in the data dictionary: the multiplying factor for the mode energy, and related quantities.
    """
    return d['M_star'] * d['R_star'] ** 2.0


def get_H_fac(d: "dict[str, Any]") -> "Any":
    """Retrieves the conversion factor for the mode enthalpy.

    Parameters
    ----------
    d : dict[str, Any]
        Data dictionary holding the necessary data to dimensionalize mode enthalpies and related quantities.

    Returns
    -------
    Any
        The requested property stored in the data dictionary: the multiplying factor for the mode enthalpy, and related quantities.
    """
    return d['G'] * d['M_star'] ** 2.0 / d['R_star']


# enumeration class for dimensionalization of GYRE mode files
class DimGyre(DimClass):
    """Enumeration (sub)class containing methods used to dimensionalize GYRE file output."""

    # enumeration functional attributes for which dimensionalization is needed
    XI_R = get_r
    XI_H = get_r
    EUL_PHI = get_eul_phi_fac
    DEUL_PHI = get_deul_phi_fac
    EUL_P = get_P
    EUL_RHO = get_rho
    EUL_T = get_T
    LAG_P = get_P
    LAG_RHO = get_rho
    LAG_T = get_T
    OMEGA_ROT = get_om_rot_fac
    # additional, unused definitions (for now)
    DELTA_P = get_om_rot_fac
    DELTA_G = get_om_rot_fac
    XI_R_REF = get_r
    XI_H_REF = get_r
    EUL_PHI_REF = get_eul_phi_fac
    DEUL_PHI_REF = get_deul_phi_fac
    LAG_S_REF = get_r
    LAG_L_REF = get_L
    LAG_S = get_r
    LAG_L = get_L
    E = get_E_fac
    E_P = get_E_fac
    E_G = get_E_fac
    H = get_H_fac
    W = get_H_fac
    W_EPS = get_H_fac
    TAU_SS = get_H_fac
    TAU_TR = get_H_fac
    DE_DX = get_E_fac
    DW_DX = get_H_fac
    DW_EPS_DX = get_H_fac
    DTAU_SS_DX = get_H_fac
    DTAU_TR_DX = get_H_fac

    # add a list containing the strings of arrays that need complex array conversion
    COMPLEX_CONVERSION_LIST = [
        'xi_r',
        'xi_h',
        'eul_phi',
        'deul_phi',
        'eul_P',
        'eul_rho',
        'eul_T',
        'xi_r_ref',
        'lag_P',
        'lag_rho',
        'lag_T',
        'omega',
        'y_1',
        'y_2',
        'y_3',
        'y_4',
        'y_5',
        'y_6',
        'freq',
        'l_i',
        'lambda',
        'omega_int',
        'dzeta_dx',
        'Yt_1',
        'Yt_2',
        'I_0',
        'I_1',
        'xi_h_ref',
        'eul_phi_ref',
        'deul_phi_ref',
        'lag_S_ref',
        'lag_L_ref',
        'lag_S',
        'lag_S_ref',
        'dbeta_dx',
        'lag_L',
    ]
    AVOID_LIST = [
        'freq_units',
        'freq_frame',
        'n',
        'j',
        'l',
        'm',
        'n_p',
        'n_g',
        'n_pg',
        'prop_type',
    ]
