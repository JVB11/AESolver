"""Python module containing enumeration superclasses that aid conversions of read-file comments to usable dictionary keys.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import logging
import sys
from enum import Enum

# type checking imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

# set up logger
logger = logging.getLogger(__name__)


# enumeration superclass used to read data
class EnumerationDefaultSuperRead(Enum):
    """Enumeration superclass that contains methods used by its subclasses to read relevant default information.

    Notes
    -----
    Enumeration attributes are defined in the subclasses!
    """

    @classmethod
    def get_relevant_attr(
        cls, attrname: str, input_val: object | None = None
    ) -> "Any":
        """Class method used to access a default attribute of the enumeration subclass, if necessary.

        Notes
        -----
        If the input value is not None, the input value shall be returned instead of that default value.

        Parameters
        ----------
        attrname : str
            The name of the attribute of the enumeration subclass.
        input_val : object or None, optional
            The input value with which the default can be replaced, if not None. If None, the default value is used. Default: None.

        Returns
        -------
        Any
            The input value if 'input_val' is None. Otherwise, the default value of the attribute with name 'attrname' is returned.
        """
        try:
            if input_val is None:
                return cls.__getitem__(attrname.upper()).value
            else:
                return input_val
        except KeyError:
            # log error message and exit
            logger.error(
                f'Enumeration attribute ({attrname}) not recognized. Now exiting.'
            )
            sys.exit()
