"""Python module containing wrapper class for the enumeration class used for the normalization of GYRE eigenfunctions.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import logging
import sys
from typing import Callable


# set up logger
logger = logging.getLogger(__name__)


# wrapper class used for the enumeration class methods
class EnumerationNormalizationMethodWrapper:
    """Wrapper class that is used to produce cleaner code for checking normalization and computing normalization factors.

    Parameters
    ----------
    my_method : Callable
        The method you are trying to wrap.
    """

    # attribute type declarations
    func: Callable
    indexer: int

    def __init__(self, my_method):
        # store the method you are wrapping
        self.func = my_method
        # decide which action to undertake based on method name
        if my_method.__name__ == 'factor_from_object':
            self.indexer = 0
        else:
            self.indexer = 1

    def __call__(self, *args, **kwargs):
        """Wrapping call for a class method. Will do common actions such as checking whether enumeration attributes exist and/or are implemented.

        Notes
        -----
        args[0] refers to the enumeration class that contains the class method that is being wrapped.

        Returns
        -------
        float or complex or bool
            Returns the output of the wrapped method.

        Raises
        ------
        my_action : NotImplementedError | Callable
            If NotImplementedError, it warns the user the method is not yet implemented. Otherwise, nothing will be raised.
        """
        # try to obtain the appropriate value from the class object
        try:
            # retrieves the appropriate action from the enumeration class: args[0] == enumeration class for the class method wrap
            my_action = (
                args[0]
                .__getitem__(kwargs['normalization_action_string'].upper())
                .value[self.indexer]
            )
        except AttributeError:
            logger.exception(
                'No known action of this enumeration class was passed.'
            )
            sys.exit()
        # try raising action to see if it is an exception
        try:
            raise my_action('This action was not (yet) implemented.')
        except TypeError:
            # return the updated function containing the action kwarg
            return self.func(*args, **kwargs, action=my_action)
