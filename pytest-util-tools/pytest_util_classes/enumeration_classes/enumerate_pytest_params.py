"""Enumeration superclass module for pytest parameters.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
from _pytest.mark import structures
import pytest
import sys
from enum import Enum

# typing imports
from typing import TYPE_CHECKING


# typing definitions
if TYPE_CHECKING:
    my_pytest_param = structures.ParameterSet
else:
    from typing import Any

    my_pytest_param = Any


# generate enumeration object containing the pytest parameters for the tests
class EnumeratePytestParameters(Enum):
    """Enumeration object containing the pytest parameters for the tests."""

    # retrieve the parameter corresponding to the input value
    @classmethod
    def get_pytest_parameter(cls, name: str) -> my_pytest_param:
        """Get the pytest parameter based on its name.

        Parameters
        ----------
        name : str
            The name of the parameter.

        Returns
        -------
        pytest.param
            The pytest parameter
        """
        try:
            if (_my_param_fail_reason := cls.__getitem__(name).value) is None:
                return pytest.param(name)
            else:
                return pytest.param(
                    name, marks=pytest.mark.xfail(reason=_my_param_fail_reason)
                )
        except AttributeError:
            print(
                f'Attribute {name} does not exist in this enumeration object! '
                f'Now exiting.'
            )
            sys.exit()

    # retrieve list of all parameters
    @classmethod
    def get_all_pytest_parameters(cls) -> list[my_pytest_param]:
        """Retrieves and returns all pytest parameters.

        Returns
        -------
        list[pytest.param]
            List of all pytest parameters.
        """
        return [
            cls.get_pytest_parameter(_pytest_param_name)
            for _pytest_param_name in cls.__members__.keys()
        ]
