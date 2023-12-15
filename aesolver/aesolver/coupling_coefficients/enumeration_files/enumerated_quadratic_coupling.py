"""Python module containing enumeration classes for 'quadratic_coupling_coefficient_rotating.py'.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import logging
import sys
from enum import Enum


# set up the logger
logger = logging.getLogger(__name__)


# enumeration class for storing the names of attributes stored in the quadratic_coupling object for MESA+GYRE data
class EnumRadial(Enum):
    """Enumeration class containing the names of attributes stored in the quadratic_coupling object."""

    # enumeration parameters
    RADIAL_FUNCTIONS = [
        '_z_1',
        '_z_2',
        '_x_z_1',
        '_z_2_over_c_1_omega',
        '_x_z_2_over_c_1_omega',
    ]
    RADIAL_GRADIENTS = [
        '_rad_der_x_z_1',
        '_rad_der_x_z_2_c_1_omega',
        '_rad_diverg',
    ]
    MISCELLANEOUS_PARAMETERS = ['_lag_L_dim']

    # class method used to access the specific named parameters
    @classmethod
    def access_parameters(cls, term: str) -> list[str]:
        """Class method used to access a specific set of named parameters, specified by 'term'.

        Parameters
        ----------
        term : str
            Defines which set of parameter names should be accessed.

        Returns
        -------
        list[str]
            Contains the specific set of named parameters.
        """
        # return the set of specified terms
        try:
            return cls.__getitem__(term.upper()).value
        except AttributeError:
            logger.exception(
                'No known attribute of this enumeration class was passed.'
            )
            sys.exit()

    # class method used to access all parameters
    @classmethod
    def access_all_parameters(cls) -> list[str]:
        """Class method used to access all named parameters.

        Returns
        -------
        list[str]
            Contains all named parameters.
        """
        return [_my_s for _my_enum in cls for _my_s in _my_enum.value]


# enumeration class for storing the names of attributes stored in the quadratic_coupling object for polytrope data
class EnumRadialPoly(Enum):
    """Enumeration class containing the names of attributes stored in the quadratic_coupling object."""

    # enumeration parameters
    RADIAL_FUNCTIONS = [
        '_z_1',
        '_z_2',
        '_x_z_1',
        '_z_2_over_c_1_omega',
        '_x_z_2_over_c_1_omega',
    ]
    RADIAL_GRADIENTS = [
        '_rad_der_x_z_1',
        '_rad_der_x_z_2_c_1_omega',
        '_rad_diverg',
    ]

    # class method used to access the specific named parameters
    @classmethod
    def access_parameters(cls, term: str) -> list[str]:
        """Class method used to access a specific set of named parameters, specified by 'term'.

        Parameters
        ----------
        term : str
            Defines which set of parameter names should be accessed.

        Returns
        -------
        list[str]
            Contains the specific set of named parameters.
        """
        # return the set of specified terms
        try:
            return cls.__getitem__(term.upper()).value
        except AttributeError:
            logger.exception(
                'No known attribute of this enumeration class was passed.'
            )
            sys.exit()

    # class method used to access all parameters
    @classmethod
    def access_all_parameters(cls) -> list[str]:
        """Class method used to access all named parameters.

        Returns
        -------
        list[str]
            Contains all named parameters.
        """
        return [_my_s for _my_enum in cls for _my_s in _my_enum.value]
