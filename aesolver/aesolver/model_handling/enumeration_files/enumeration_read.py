"""'Python enumeration module containing an enumeration class that aids file reading efforts for specific input.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# ensure type annotations are not evaluated
# but instead are handled as strings
from __future__ import annotations

# import statements
import logging
import sys
from enum import Enum
from operator import attrgetter

# type checking imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Protocol
    from collections.abc import Callable
    from ..read_models import ModelReader
    # define typing protocol for the reader call
    class MyReadCall(Protocol):
        def __call__(self, file_path: str, attributes: list[str]) -> dict[str, Any]:
            ...
    

# set up logger
logger = logging.getLogger(__name__)


# enumeration class used for reading information in files
class EnumerationReadInfo(Enum):
    """Enumeration class used to support reading/extracting information from GYRE/MESA files, differentiating between methods based on suffix."""

    # enumeration attributes
    H5 = attrgetter('_hdf5_reader')
    HDF5 = attrgetter('_hdf5_reader')
    TXT = attrgetter('_text_reader')
    DAT = attrgetter('_dat_reader')

    # return the appropriate reader method
    @classmethod
    def reader(
        cls, file_suffix: str, file_path: str
    ) -> "Callable[[ModelReader], MyReadCall]":
        """Class method used to return the appropriate reader method for extraction of information from GYRE/MESA files.

        Parameters
        ----------
        file_suffix : str
            The file suffix based on which the appropriate reader method is selected.
        file_path : str
            Path to the file whose suffix is passed in 'file_suffix'.

        Returns
        -------
        Callable[[ModelReader], MyReadCall]
            The method used to select the correct information reading/extraction method.
        """
        # retrieve and return the appropriate method, if possible
        try:
            return getattr(cls, file_suffix.upper()).value
        except AttributeError:
            logger.exception(
                f'No known/supported file suffix found for the file with path {file_path}. Now exiting.'
            )
            sys.exit()
