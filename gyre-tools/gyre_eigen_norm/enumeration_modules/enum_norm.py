"""Python enumeration module containing enumeration class and its wrapper for actions undertaken during normalization.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import logging
import sys
import numpy as np
from enum import Enum

# custom local wrapper module import
from .enum_norm_wrapper import EnumerationNormalizationMethodWrapper as ENMw


# set up logger
logger = logging.getLogger(__name__)


# enumeration class that computes the multiplicative normalization factors for the different normalization conventions
class EnumNormConvFactors(Enum):
    """Python enumeration class that computes the multiplicative normalization factors for (known) normalization conventions."""

    # enumeration attributes
    # --> construct normalization factor from object attributes NOTE: non-object actions not yet implemented!
    # first in action tuple attributes = object-based actions
    # second in action_tuple attributes = object-based normalization energy factor actions
    # third in action_tuple attributes = non-object-based actions
    # fourth in action_tuple attributes = non-object-based normalization energy factor actions
    LEE2012 = (
        lambda nr, obj: np.sqrt(
            (obj._G * obj._M_star ** (2.0))
            / (obj._R_star * getattr(obj, f'_energy_mode_{nr}'))
        ),
        (
            lambda obj: (obj._G * obj._M_star ** (2.0)) / obj._R_star,
            lambda nr, obj: getattr(obj, f'_energy_mode_{nr}'),
        ),
        NotImplementedError,
        NotImplementedError,
    )
    SCHENK2002 = (
        lambda nr, obj: np.sqrt(
            (obj._M_star * obj._R_star ** (2.0))
            / getattr(obj, f'_classic_norm_mode_{nr}')
        ),
        (
            lambda obj: obj._M_star * obj._R_star ** (2.0),
            lambda nr, obj: getattr(obj, f'_classic_norm_mode_{nr}'),
        ),
        NotImplementedError,
        NotImplementedError,
    )

    # compute the normalization factor based on input from object
    @classmethod
    @ENMw
    def factor_from_object(
        cls, my_obj, normalization_action_string, my_mode_nr, action=None
    ) -> float | complex:
        """Compute the normalization factor, based on attributes stored in object.

        Notes
        -----
        The wrapper class 'EnumerationNormalizationMethodWrapper' will retrieve the appropriate action from the enumeration class.

        Parameters
        ----------
        my_obj : QuadraticCouplingCoefficientRotating or QCCR
            The object in which the attributes are stored which need to be used to compute the normalization factor.
        normalization_action_string : str
            The string indicating which normalization factor is to be computed.
        my_mode_nr : int
            The number of the mode.
        action : None or Callable, optional
            The action method/function returned by the decorator, if possible; by default None.

        Returns
        -------
        float | complex
            The normalization factor for the eigenfunction, based on stored object information.
        """
        # compute the normalization factor and return it
        if action is None:
            logger.error('No action passed. Now exiting.')
            sys.exit()
        else:
            return action(my_mode_nr, my_obj)

    # check the values for the normalization based on input from object
    @classmethod
    @ENMw
    def check_value_from_object(
        cls, my_obj, normalization_action_string, my_mode_nr, action=None
    ) -> bool:
        """Compare the values useful for the normalization based on input from object with the check values.

        Notes
        -----
        The wrapper class 'EnumerationNormalizationMethodWrapper' will retrieve the appropriate action from the enumeration class.

        Parameters
        ----------
        my_obj : QuadraticCouplingCoefficientRotating | QCCR
            The object in which the attributes are stored which need to be used to compute the normalization factor.
        normalization_action_string : str
            The string indicating which normalization factor is to be computed.
        my_mode_nr : int
            The number of the mode.
        action : None or Callable, optional
            The action method/function returned by the decorator, if possible; by default None.

        Returns
        -------
        bool
            True if additional normalization steps are required. False if the normalization is ok.
        """
        # make the comparison and return its result
        if action is None:
            logger.error('Passed action is None. Now exiting.')
            sys.exit()
        else:
            return action[0](my_obj) == action[1](my_mode_nr, my_obj)

    # get percentage difference of values based on input from object
    @classmethod
    @ENMw
    def proc_diff_from_object(
        cls, my_obj, normalization_action_string, my_mode_nr, action=None
    ):
        """Compute the percentage differences for the normalization based on input from object.

        Notes
        -----
        The wrapper class 'EnumerationNormalizationMethodWrapper' will retrieve the appropriate action from the enumeration class.

        Parameters
        ----------
        my_obj : QuadraticCouplingCoefficientRotating or QCCR
            The object in which the attributes are stored which need to be used to compute the normalization factor.
        normalization_action_string : str
            The string indicating which normalization factor is to be computed.
        my_mode_nr : int
            The number of the mode.
        action : None or Callable, optional
            The action method/function returned by the decorator, sif possible, by default None.

        Returns
        -------
        float
            Percentage difference of normalization.
        """
        # compute the difference and return the result
        if action is None:
            logger.error('Passed action is None. Now exiting.')
            sys.exit()
        else:
            _diff = action[0](my_obj) - action[1](my_mode_nr, my_obj)
            return (_diff / action[0](my_obj)) * 100.0
