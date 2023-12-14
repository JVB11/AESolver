"""Defines a coupler enumeration class to be used for testing. Mock-up of the actual coupler class.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import json
from enum import Enum


# store the mockup enumeration class
class MockEnumCoupler(Enum):
    """Enumeration class that stores attributes used in coupling calculations."""

    # enumeration parameters
    RADIAL_FUNCTIONS = json.dumps(
        [
            '_z_1',
            '_z_2',
            '_x_z_1',
            '_z_2_over_c_1_omega',
            '_x_z_2_over_c_1_omega',
        ]
    )
    RADIAL_GRADIENTS = json.dumps(
        ['_rad_der_x_z_1', '_rad_der_x_z_2_c_1_omega', '_rad_diverg']
    )
    MISCELLANEOUS_PARAMETERS = json.dumps(['_lag_L_dim'])

    # class method used to access specific named parameters
    @classmethod
    def access_parameters(cls, term):
        """Class method used to access a specific set of parameter names, specified by 'term'.

        Parameters
        ----------
        term : str
            Defines which set of parameter names should be accessed.

        Returns
        -------
        list[str]
            Contains the specific set of parameter names.
        """
        # return the set of specific terms WITHOUT ERROR HANDLING
        return json.loads(cls.__getitem__(term.upper()).value)

    # class method used to access all parameters
    @classmethod
    def access_all_parameters(cls):
        """Class method used to access all parameter names.

        Returns
        -------
        list[str]
            Contains all parameter names.
        """
        return [
            _my_s
            for _my_key in cls.__members__.keys()
            for _my_s in cls.access_parameters(_my_key)
        ]
