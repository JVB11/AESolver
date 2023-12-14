"""Python module containing classes used to perform unit conversions for GYRE output data.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import logging
from enum import Enum


# set up logger
logger = logging.getLogger(__name__)


# enumeration class that holds unit conversions for GYRE frequency units
class GYREFreqUnitConverter(Enum):
    """Enumeration class containing conversions for GYRE frequency units."""

    # enumeration attributes
    CYC_PER_DAY = r'd$^{-1}$'
    HZ = r'Hz'
    UHZ = r'$\mu$Hz'
    RAD_PER_SEC = r'rad s$^{-1}$'

    # get the correct enumeration attribute translation
    @classmethod
    def convert(cls, text):
        """Class method converting 'text' into corresponding enumeration attribute, if possible.

        Parameters
        ----------
        text : str
            Value of the GYRE 'freq_units' parameter.

        Returns
        -------
        str
            Translation of the GYRE 'freq_units' parameter.
        """
        try:
            return cls.__getitem__(
                text.decode() if isinstance(text, bytes) else text
            ).value
        except KeyError:
            logger.info(
                f"Key '{text}' not found, now returning no unit. (This may be wanted behavior.)"
            )
            return ''
