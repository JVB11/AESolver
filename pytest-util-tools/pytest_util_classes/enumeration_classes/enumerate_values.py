"""Factory module used to enumerate the common input and
expected output values for tests.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import sys
import json
import numpy as np
from enum import Enum

# typing imports
import logging

# type checking imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy.typing as npt


logger = logging.getLogger(__name__)


# factory class for enumeration of tests
class EnumeratedValues(Enum):
    """Enumeration class used to store common input
    and expected output values for tests.
    """

    @classmethod
    def get_value(
        cls, my_key: str, dtype_buffered: 'npt.DTypeLike | None' = None
    ) -> int | float | np.ndarray | list | str:
        """Generic getter method used to retrieve
        buffered and non-buffered values of
        enumeration members.

        Parameters
        ----------
        my_key : str
            The enumeration member name.
        dtype_buffered : npt.DTypeLike | None, optional
            If None, ignore this parameter.
            If not None, use this parameter to specify a
            specific buffered numpy array type.
            Default value: None.

        Returns
        -------
        int | float | np.ndarray | list | str
        """
        # attempt to retrieve the value of an enumeration member
        try:
            # attempt to retrieve buffered item (e.g. numpy ndarray)
            if dtype_buffered:
                return np.frombuffer(
                    my_val := cls.__getitem__(my_key.upper()).value,
                    dtype=dtype_buffered,
                )
            else:
                return np.frombuffer(
                    my_val := cls.__getitem__(my_key.upper()).value
                )
        except KeyError:
            print(
                f'Unknown member of enumeration class ({my_key}). '
                f'Now exiting.'
            )
            sys.exit()
        except TypeError:
            try:
                # attempt to retrieve json-dumped item (e.g. list)
                return json.loads(my_val)  # type: ignore
            except UnboundLocalError:
                logger.exception(
                    'Could not retrieve a value from this enumeration class (would be called "my_val" in this class method). Now exiting.'
                )
                sys.exit()
            except (TypeError, json.JSONDecodeError):
                # return the value without adjustment (e.g. float or int or str)
                return my_val  # type: ignore

    @classmethod
    def get_enumeration_dict(cls) -> dict:
        """Retrieves all stored key-value pairs of enumeration members.

        Returns
        -------
        dict
            The values dictionary containing the relevant key-value pairs.
        """
        return {
            _key.lower(): cls.get_value(_key) for _key in cls.__members__.keys()
        }
