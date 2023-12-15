"""Python enumeration module that contains an enumeration objects that store data used to obtain the frequencies in the correct units for computation of mode energies, and mode eigenfunction normalization.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# ensure type annotations are not evaluated
# but instead are handled as strings
from __future__ import annotations

# import statements
import logging
import sys
import numpy as np
from enum import Enum

# typing imports
from collections.abc import Callable
from typing import TYPE_CHECKING

# import classes for type checking
if TYPE_CHECKING:
    from ...coupling_coefficients import QCCR


# set up logger
logger = logging.getLogger(__name__)


# function that determines whether a specific frequency unit is angular or not
def is_angular_frequency(requested_frequency_unit: str) -> bool:
    """Function determining whether a requested frequency unit is angular or not.

    Parameters
    ----------
    requested_frequency_unit : str
        Requested frequency unit.

    Returns
    -------
    bool
        If True, the requested frequency is angular. If False, it is cyclic.
    """
    # get lower-case string of requested frequency unit
    lowered_unit = requested_frequency_unit.lower()
    # check if it is angular and return the resulting boolean
    return (
        (lowered_unit == 'rps')
        or (lowered_unit == 'rad_per_sec')
        or (lowered_unit == 'rpd')
        or (lowered_unit == 'rad_per_day')
    )


# define the class that performs the frequency conversion if necessary
class EnumerationGYREFreqConverter(Enum):
    """Enumeration class containing information used to perform the frequency conversion actions necessary for computation of mode energies.

    Notes
    -----
    Necessary frequency units for computation of mode energies are in Hz (CGS UNITS!). This is only implemented for 'NONE', 'HZ', 'UHZ', 'RAD_PER_SEC', 'CYC_PER_DAY'. Using other GYRE units will not return the correct result!!!
    """

    # enumeration attributes == GYRE frequency unit strings (also works for rotation frequency)
    # --> points to function tuple if necessary for conversion, else None
    # --> first function/None: conversion from object
    # --> second function/None: conversion from array and function input
    NONE = (
        lambda val, obj: val
        * np.sqrt(obj._G * obj._M_star / (obj._R_star ** (3.0))),
        None,
    )
    HZ = (None, None)
    UHZ = (lambda val, _: val / 1.0e6, None)
    RAD_PER_SEC = (lambda val, _: val / (2.0 * np.pi), None)
    CYC_PER_DAY = (lambda val, _: val / 86400.0, None)
    # NOTE: not (yet) implemented
    ACOUSTIC_DELTA = (
        NotImplementedError('This action was not (yet) implemented'),
        NotImplementedError('This action was not (yet) implemented'),
    )
    GRAVITY_DELTA = (
        NotImplementedError('This action was not (yet) implemented'),
        NotImplementedError('This action was not (yet) implemented'),
    )
    UPPER_DELTA = (
        NotImplementedError('This action was not (yet) implemented'),
        NotImplementedError('This action was not (yet) implemented'),
    )
    LOWER_DELTA = (
        NotImplementedError('This action was not (yet) implemented'),
        NotImplementedError('This action was not (yet) implemented'),
    )
    ACOUSTIC_CUTOFF = (
        NotImplementedError('This action was not (yet) implemented'),
        NotImplementedError('This action was not (yet) implemented'),
    )
    GRAVITY_CUTOFF = (
        NotImplementedError('This action was not (yet) implemented'),
        NotImplementedError('This action was not (yet) implemented'),
    )
    ROSSBY_I = (
        NotImplementedError('This action was not (yet) implemented'),
        NotImplementedError('This action was not (yet) implemented'),
    )
    ROSSBY_O = (
        NotImplementedError('This action was not (yet) implemented'),
        NotImplementedError('This action was not (yet) implemented'),
    )
    CRITICAL = (
        NotImplementedError('This action was not (yet) implemented'),
        NotImplementedError('This action was not (yet) implemented'),
    )
    # list of mapping dictionaries per requested unit (or Notimplementederror)
    NONE_REQUESTED = NotImplementedError(
        'This action was not (yet) implemented'
    )
    HZ_REQUESTED = {
        b'HZ': lambda val: val,
        b'UHZ': lambda val: val * 1e-6,
        b'RAD_PER_SEC': lambda val: val / (2.0 * np.pi),
        b'CYC_PER_DAY': lambda val: val / 86400.0,
    }
    UHZ_REQUESTED = {
        b'HZ': lambda val: val * 1e6,
        b'UHZ': lambda val: val,
        b'RAD_PER_SEC': lambda val: val * 1e6 / (2.0 * np.pi),
        b'CYC_PER_DAY': lambda val: val * 1e6 / 86400.0,
    }
    CYC_PER_DAY_REQUESTED = {
        b'HZ': lambda val: val * 86400.0,
        b'UHZ': lambda val: val * 1e-6 * 86400.0,
        b'RAD_PER_SEC': lambda val: val * 86400.0 / (2.0 * np.pi),
        b'CYC_PER_DAY': lambda val: val,
    }
    CPD_REQUESTED = {
        b'HZ': lambda val: val * 86400.0,
        b'UHZ': lambda val: val * 1e-6 * 86400.0,
        b'RAD_PER_SEC': lambda val: val * 86400.0 / (2.0 * np.pi),
        b'CYC_PER_DAY': lambda val: val,
    }  # alias for cycles per day
    RAD_PER_SEC_REQUESTED = {
        b'HZ': lambda val: val * (2.0 * np.pi),
        b'UHZ': lambda val: val * 1e-6 * (2.0 * np.pi),
        b'RAD_PER_SEC': lambda val: val,
        b'CYC_PER_DAY': lambda val: val * (2.0 * np.pi) / 86400.0,
    }
    ACOUSTIC_DELTA_REQUESTED = NotImplementedError(
        'This action was not (yet) implemented'
    )
    GRAVITY_DELTA_REQUESTED = NotImplementedError(
        'This action was not (yet) implemented'
    )
    UPPER_DELTA_REQUESTED = NotImplementedError(
        'This action was not (yet) implemented'
    )
    LOWER_DELTA_REQUESTED = NotImplementedError(
        'This action was not (yet) implemented'
    )
    ACOUSTIC_CUTOFF_REQUESTED = NotImplementedError(
        'This action was not (yet) implemented'
    )
    GRAVITY_CUTOFF_REQUESTED = NotImplementedError(
        'This action was not (yet) implemented'
    )
    ROSSBY_I_REQUESTED = NotImplementedError(
        'This action was not (yet) implemented'
    )
    ROSSBY_O_REQUESTED = NotImplementedError(
        'This action was not (yet) implemented'
    )
    CRITICAL_REQUESTED = NotImplementedError(
        'This action was not (yet) implemented'
    )

    # convert the frequency based on enumeration attribute: STR MODE
    @classmethod
    def factor_from_str(
        cls, my_units: bytes, my_requested_units: str, no_exit: bool = False
    ) -> Callable | KeyError:
        """Generate frequency conversion factor for the GYRE frequencies, based on the supplied units string and the requested units string.

        Parameters
        ----------
        my_units : str
            Denotes the current units.
        my_requested_units : str
            Denotes the requested units.
        no_exit : bool, optional
            If True, do not allow the sys.exit() call. If False, allow this call; by default False.

        Returns
        -------
        Callable | KeyError
            The multiplicative conversion factor.
        """
        # retrieve the mapping dictionary for frequency conversion
        func_dict = cls.__getitem__(
            f'{my_requested_units.upper()}_REQUESTED'
        ).value
        # return the conversion function
        try:
            return func_dict[my_units]  # type: ignore
        except KeyError as ke:
            # allow exception raise
            if not no_exit:
                logger.exception('Unknown key provided. Now exiting.')
                sys.exit()
            else:
                # return an error
                return ke

    # convert the frequency based on enumeration attribute: DICT MODE
    @classmethod
    def factor_from_dict(
        cls, my_mode_dict: dict, my_requested_units: str, no_exit: bool = False
    ) -> Callable:
        """Perform frequency conversion of the GYRE frequency, based on the supplied mode dictionary and the requested units string.

        Notes
        -----
        This mode can only be used for implemented methods -- see class header.

        Parameters
        ----------
        my_mode_dict : dict
            Contains the necessary mode information to retrieve the mode frequency.
        my_requested_units : str
            Denotes the units in which you want your frequency to be.
        no_exit : bool, optional
            If True, do not allow the sys.exit() call. If False, allow this call; by default False.

        Returns
        -------
        Callable
            The function used to retrieve the mode frequency.
        """
        # get the strings denoting the units
        _unit_string: bytes = my_mode_dict['freq_units']
        # return the conversion function
        try:
            # get the factor function
            factor_func = cls.factor_from_str(
                _unit_string, my_requested_units, no_exit=no_exit
            )
            if TYPE_CHECKING:
                assert isinstance(factor_func, KeyError)
            # check if an error is returned
            raise factor_func
        except TypeError:
            # get the factor function
            factor_func = cls.factor_from_str(
                _unit_string, my_requested_units, no_exit=no_exit
            )
            if TYPE_CHECKING:
                assert not isinstance(factor_func, KeyError)
            # error was not returned: cannot raise --> output OK
            return factor_func
        except KeyError:
            # handle cases where output is not OK --> b'NONE' (i.e. dimensionless omegas!!!)
            # - retrieve dimensioning factors
            _G = my_mode_dict['G']
            _M = my_mode_dict['M_star']
            _R = my_mode_dict['R_star']
            # get additional factor from requested units
            _req_fac = cls.factor_from_str(b'RAD_PER_SEC', my_requested_units)
            # assert _req_fac is OK
            if TYPE_CHECKING:
                assert not isinstance(_req_fac, KeyError)
            # return the dimensioning function
            return lambda val: _req_fac(val * np.sqrt((_G * _M) / (_R**3.0)))

    # convert the frequency based on enumeration attribute: OBJECT MODE
    @classmethod
    def factor_from_object(
        cls,
        my_obj: 'QCCR',
        my_freq_base_string: str,
        my_mode_nr: int,
        my_index: int | None = None,
        use_complex: bool = False,
    ) -> float:
        """Perform frequency conversion of the GYRE frequency, based on its frequency_units string, and a supplied object.

        Notes
        -----
        This mode can only be used for implemented methods -- see class header.

        Parameters
        ----------
        my_obj : QCCR
            The coupling object containing necessary information.
        my_freq_base_string : str
            The string describing the quantity/attribute of the object that represents the frequency.
        my_mode_nr : int
            The number of the mode.
        my_index : int | None, optional
            The (integer) index of the quantity, if necessary. None if not necessary; by default None.
        use_complex : bool, optional
            If True, return a complex-valued frequency. If False, return a real-valued frequency; by default False.

        Returns
        -------
        my_frequency : float
            The converted frequency value.
        """
        # based on the base frequency string, retrieve the frequency units
        unit_string = getattr(
            my_obj, f'{my_freq_base_string}_units_mode_{my_mode_nr}'
        )
        # based on the frequency units, retrieve the appropriate action
        try:
            # convert bytes-string to unicode string, if necessary
            if isinstance(unit_string, bytes):
                unit_string = unit_string.decode()
            # retrieve the action operator for the object
            my_act = cls.__getitem__(unit_string.upper()).value
            if TYPE_CHECKING:
                assert isinstance(my_act, tuple)
            my_action = my_act[0]
            if TYPE_CHECKING:
                assert isinstance(my_action, Callable | NotImplementedError)
        except AttributeError:
            logger.exception(
                'No known action of this enumeration class was passed.'
            )
            sys.exit()
        # try raising the action to see if it is an exception
        try:
            if TYPE_CHECKING:
                assert isinstance(my_action, NotImplementedError)
            raise my_action
        except TypeError:
            # no exception raised, so continue on by performing the action
            # and checking if the action needs indexing
            # - get frequency (complex- or real-valued)
            my_frequency = getattr(
                my_obj, f'{my_freq_base_string}_mode_{my_mode_nr}'
            )
            if not use_complex:
                my_frequency = my_frequency.real
            # - perform the action on the frequency
            if TYPE_CHECKING:
                assert isinstance(my_action, Callable)
            my_frequency = my_action(my_frequency, my_obj)
            # - perform indexing if needed
            if my_index is not None:
                my_frequency = my_frequency[my_index]
            # return the frequency that is converted to its required value
            return my_frequency

    @classmethod
    def factor_for_dimensioning_factor(
        cls, my_requested_units: str
    ) -> Callable:
        """Perform frequency conversion of the dimensioning factor, based on the supplied mode dictionary and the requested units string.

        Notes
        -----
        This mode can only be used for implemented methods -- see class header.

        Parameters
        ----------
        my_mode_dict : dict
            Contains the necessary mode information to retrieve the mode frequency.
        my_requested_units : str
            Denotes the units in which you want your frequency to be.
        angular : bool
            If True, an angular frequency is requested. If False, a cyclic frequency is requested.

        Returns
        -------
        Callable
            The function used to convert the dimensioning factor.
        """
        # verify the requested frequency is angular or cyclic
        is_angular = is_angular_frequency(
            requested_frequency_unit=my_requested_units
        )
        # get the conversion factor
        _req_fac = cls.factor_from_str(b'HZ', my_requested_units)
        if isinstance(_req_fac, KeyError):
            logger.error('Unknown requested units. Now exiting.')
            sys.exit()
        # compose the dimensioning factor conversion function
        if is_angular:
            # the conversion factor needs to be in the correct cyclic frequency unit: therefore one needs to divide by the nr. of radians included in the angular frequency conversion factor
            return lambda dim_fac: _req_fac(dim_fac) / (2.0 * np.pi)
        else:
            # no conversion needs to be done with respect to angularity of the frequency (conversion only needs to take care of difference between Hz and cpd)
            return _req_fac
