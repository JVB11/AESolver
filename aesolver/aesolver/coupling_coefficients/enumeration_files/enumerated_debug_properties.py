"""File containing enumeration classes used to list debug properties.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
from enum import Enum


# enumeration class listing min_max debug properties for the CouplingCoefficientRotating class
class DebugMinMaxCouplingCoefficientRotating(Enum):
    """Contains the debug property names for the min_max debug properties used for the CouplingCoefficientRotating class."""

    # define the enumeration attributes
    z_1 = ()
    z_2 = ()
    x_z_1 = ()
    z_2_over_c_1_omega = ()
    x_z_2_over_c_1_omega = ()
    rad_der_x_z_1 = ()
    rad_der_x_z_2_c_1_omega = ()
    rad_diverg = ()

    # retrieve the list of attributes for the tests
    @classmethod
    def get_mode_specific_attrs(cls, mode_nr: int = 1) -> list[str]:
        """Returns the debug attributes for the min_max debug properties for the CouplingCoefficientRotating class.

        Parameters
        ----------
        mode_nr : int, optional
            The nr identifying the mode in the considered triad.

        Returns
        -------
        attr_list : list[str]
            Returns the min_max debug properties of the CouplingCoefficientRotating class.
        """
        # initialize the attribute list
        attr_list = []
        # fill up the attribute list
        for _my_enum_attr in cls:
            # obtain the base attribute name
            _base_name = _my_enum_attr.name
            # add mode specific related attribute
            attr_list.append(f'_{_base_name}_mode_{mode_nr}_max_min')
        # return the attribute list
        return attr_list


# enumeration class listing min_max debug properties for the QuadraticCouplingCoefficient class
class DebugMinMaxQuadraticCouplingCoefficientRotating(Enum):
    """Contains the debug property names for the min_max debug properties used for the QuadraticCouplingCoefficientRotating class."""

    # define the enumeration attributes
    hr = ()
    ht = ()
    hp = ()
    theta_der_hr = ()
    theta_der_ht = ()
    theta_der_hp = ()
    theta_phi = ()
    phi_phi = ()
    mhr = ()
    theta_der2_hr = ()
    theta_der_hr_theta = ()
    theta_der2_hr_theta = ()
    theta_der_ht_theta = ()
    theta_der_hp_theta = ()

    # retrieve the list of attributes for the tests
    @classmethod
    def get_mode_specific_attrs(cls, mode_nr: int = 1) -> list[str]:
        """Returns the debug attributes for the min_max debug properties for the QuadraticCouplingCoefficientRotating class.

        Parameters
        ----------
        mode_nr : int, optional
            The nr identifying the mode in the considered triad.

        Returns
        -------
        attr_list : list[str]
            Returns the min_max debug properties of the QuadraticCouplingCoefficientRotating class.
        """
        # initialize the attribute list
        attr_list = []
        # fill up the attribute list
        for _my_enum_attr in cls:
            # obtain the base attribute name
            _base_name = _my_enum_attr.name
            # add Hough function related attribute
            attr_list.append(f'_{_base_name}_mode_{mode_nr}_max_min')
            # add a normalized version
            attr_list.append(f'_{_base_name}_mode_{mode_nr}_max_min_normed')
        # return the attribute list
        return attr_list


# enumeration class listing min_max debug properties for the SuperCouplingCoefficient class
class DebugMinMaxSuperCouplingCoefficient(Enum):
    """Contains the debug property names for the min_max debug properties used for the SuperCouplingCoefficient class."""

    # define the enumeration attributes
    dgravpot = ()
    dgravpot2 = ()
    dgravpot3 = ()
    P = ()
    dP = ()
    dP2 = ()
    dP3 = ()
    rho = ()
    drho = ()
    drho2 = ()
    drho3 = ()

    # retrieve the list of attributes for the tests
    @classmethod
    def get_attrs(cls) -> list[str]:
        """Returns the debug attributes for the min_max debug properties for the SuperCouplingCoefficient class.

        Returns
        -------
        attr_list : list[str]
            Returns the min_max debug properties of the SuperCouplingCoefficient class.
        """
        # initialize the attribute list
        attr_list = []
        # fill up the attribute list
        for _my_enum_attr in cls:
            # obtain the base attribute name
            _base_name = _my_enum_attr.name
            # add regular attribute
            attr_list.append(f'_{_base_name}_max_min')
            # add normed attribute
            attr_list.append(f'_{_base_name}_max_min_normed')
            # add polytrope attribute
            attr_list.append(f'_{_base_name}_max_min_poly')
            # add polytrope normed attribute
            attr_list.append(f'_{_base_name}_max_min_normed_poly')
        # return the attribute list
        return attr_list
