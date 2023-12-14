"""Python enumeration file that contains the options for the instructions for the form of the Hough functions.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import logging
import sys
from enum import Enum


# set up logger
logger = logging.getLogger(__name__)


# enumeration class containing the options for the different instructions for the forms of the Hough functions
class InstructionsEnum(Enum):
    """Enumeration class containing the options for the different instructions for the forms of the Hough functions."""

    # enumeration fields
    PRAT = {'adjust_phi_sign': False, 'sin_factor': False}
    LEE_SAIO_EQUIVALENT = {'adjust_phi_sign': True, 'sin_factor': False}
    TOWNSEND = {'adjust_phi_sign': True, 'sin_factor': True}

    # retrieve the enumeration field value of choice
    @classmethod
    def get_instructions(cls, instruction):
        """Class method used to retrieve the name of the correct instructions for the form of the Hough functions.

        Parameters
        ----------
        instruction : str
            Defines the type of instructions to be used for retrieving the correct form. Options are: ['prat', 'lee_saio_equivalent', 'townsend'].

        Returns
        -------
        dict
            The instructions dictionary.
        """
        # attempt to retrieve the instructions dictionary and return the dictionary
        try:
            return cls.__getitem__(instruction.upper()).value
        except KeyError:
            # log error message and exit
            logger.error(
                f'Instructions/definition ({instruction}) not recognized. Now exiting.'
            )
            sys.exit()
