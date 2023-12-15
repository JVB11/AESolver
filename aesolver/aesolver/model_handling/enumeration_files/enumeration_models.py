"""Python file that contains enumeration classes that depict the information stored in the MESA/GYRE files.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
from enum import Enum
from functools import partial
from itertools import chain
from operator import mul

# type checking imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # import for typing
    from typing import Any
    from collections.abc import Callable


# base enumeration class containing the functions to be used for reading data from files
class BaseFileInfo(Enum):
    """Base enumeration class containing the methods used to retrieve the possible information stored in GYRE/MESA files."""

    # internal method used to provide input as output mapping
    @staticmethod
    def _return_input(input: "Any") -> "Any":
        """Dummy internal method used to return the same as the input, if no unit conversion action is required.

        Parameters
        ----------
        input : Any
            The generic input.

        Returns
        -------
        Any
            Output that is the same as the input, if no unit conversion is required.
        """
        return input

    # method used to obtain the list of different attributes stored in the files
    @classmethod
    def attrs(cls) -> list[str]:
        """Method that retrieves the names of the attributes (keywords) stored in the files.

        Returns
        -------
        list[str]
            The list containing the names of the attributes stored in the files.
        """
        # compile the list of lists of attributes stored in the files
        list_keywords = [
            list(cls.__getitem__(_attr).value.keys())
            for _attr in cls.__members__
        ]
        # return the list of attributes stored in the files
        return list(chain.from_iterable(list_keywords))

    # method used to obtain unit conversion factors if necessary
    @classmethod
    def unit_conversions(
        cls, unit_system: str = 'SI'
    ) -> "dict[str, Callable[[Any], Any] | Callable[[float], float]]":
        """Method that retrieves the unit conversion factors for the different attributes stored in the files.

        Parameters
        ----------
        unit_system: str, optional
            The unit system to be used for possible unit conversion; by default 'SI'. Supported values are:

            * 'SI': provide unit conversion according to the Système Internationale d'unités (SI).
            * 'CGS': no unit conversions needed, as MESA/GYRE files are in cgs units.

        Returns
        -------
        dict[str, Callable[[Any] | Any] | Callable[[float] | float]]
            The dictionary containing the functional mappings of the values with possible unit conversion.
        """
        # compile the dictionary containing the mappings of the values to the unit-converted values, and return it
        return {
            _key: cls._return_input
            if (_my_action := _val.get(unit_system)) is None
            else partial(mul, _my_action)
            for _attr in cls.__members__
            for _key, _val in cls.__getitem__(_attr).value.items()
        }


# class containing the information stored in the GYRE detail files + unit conversion
class GYREDetailFiles(BaseFileInfo):
    """Enumeration subclass containing the possible information stored in the GYRE detail files, as well as unit conversion. Inherits the methods from the BaseFileInfo class."""

    # define the elements + their unit conversion methods
    SOLUTION_DATA = {
        'n': {'SI': None},
        'omega': {'SI': None},
        'x': {'SI': None},
        'y_1': {'SI': None},
        'y_2': {'SI': None},
        'y_3': {'SI': None},
        'y_4': {'SI': None},
        'y_5': {'SI': None},
        'y_6': {'SI': None},
    }
    OBSERVABLES = {
        'freq': {'SI': None},
        'freq_units': {'SI': None},
        'freq_frame': {'SI': None},
        'f_T': {'SI': None},
        'f_g': {'SI': None},
        'psi_T': {'SI': None},
        'psi_g': {'SI': None},
    }
    CLASSIFICATION_VALIDATION = {
        'j': {'SI': None},
        'l': {'SI': None},
        'l_i': {'SI': None},
        'm': {'SI': None},
        'n_p': {'SI': None},
        'n_g': {'SI': None},
        'n_pg': {'SI': None},
        'omega_int': {'SI': None},
        'dzeta_dx': {'SI': None},
        'Yt_1': {'SI': None},
        'Yt_2': {'SI': None},
        'I_0': {'SI': None},
        'I_1': {'SI': None},
        'prop_type': {'SI': None},
    }
    PERTURBATIONS = {
        'x_ref': {'SI': None},
        'xi_r_ref': {'SI': None},
        'xi_h_ref': {'SI': None},
        'eul_phi_ref': {'SI': None},
        'deul_phi_ref': {'SI': None},
        'lag_S_ref': {'SI': None},
        'lag_L_ref': {'SI': None},
        'xi_r': {'SI': None},
        'xi_h': {'SI': None},
        'eul_phi': {'SI': None},
        'deul_phi': {'SI': None},
        'lag_S': {'SI': None},
        'lag_L': {'SI': None},
        'eul_P': {'SI': None},
        'eul_rho': {'SI': None},
        'eul_T': {'SI': None},
        'lag_P': {'SI': None},
        'lag_rho': {'SI': None},
        'lag_T': {'SI': None},
    }
    ENERGETICS_TRANSPORT = {
        'eta': {'SI': None},
        'E': {'SI': None},
        'E_p': {'SI': None},
        'E_g': {'SI': None},
        'E_norm': {'SI': None},
        'E_ratio': {'SI': None},
        'H': {'SI': None},
        'W': {'SI': None},
        'W_eps': {'SI': None},
        'tau_ss': {'SI': None},
        'tau_tr': {'SI': None},
        'dE_dx': {'SI': None},
        'dW_dx': {'SI': None},
        'dW_eps_dx': {'SI': None},
        'dtau_ss_dx': {'SI': None},
        'dtau_tr_dx': {'SI': None},
        'alpha_0': {'SI': None},
        'alpha_1': {'SI': None},
    }
    ROTATION = {
        'domega_rot': {'SI': None},
        'dfreq_rot': {'SI': None},
        'beta': {'SI': None},
        'dbeta_dx': {'SI': None},
        'lambda': {'SI': None},
    }
    STELLAR_STRUCTURE = {
        'M_star': {'SI': 1.0e-3},
        'R_star': {'SI': 1.0e-2},
        'L_star': {'SI': 1.0e-7},
        'Delta_p': {'SI': None},
        'Delta_g': {'SI': None},
        'V_2': {'SI': None},
        'As': {'SI': None},
        'U': {'SI': None},
        'c_1': {'SI': None},
        'Gamma_1': {'SI': None},
        'nabla': {'SI': None},
        'nabla_ad': {'SI': None},
        'dnabla_ad': {'SI': None},
        'upsilon_r': {'SI': None},
        'c_lum': {'SI': None},
        'c_rad': {'SI': None},
        'c_thn': {'SI': None},
        'c_thk': {'SI': None},
        'c_eps': {'SI': None},
        'kap_rho': {'SI': None},
        'kap_T': {'SI': None},
        'eps_rho': {'SI': None},
        'eps_T': {'SI': None},
        'Omega_rot': {'SI': None},
        'M_r': {'SI': 1.0e-3},
        'P': {'SI': 1.0e-1},
        'rho': {'SI': 1.0e3},
        'T': {'SI': None},
    }

    # method used to obtain the mode-independent attributes
    @classmethod
    def retrieve_mode_independent_names(cls, upper: bool = False) -> list[str]:
        """Overloaded Method used to retrieve a list of the names of the attributes that are common among modes.

        Parameters
        ----------
        upper : bool, optional
            If True, capitalize all letters of the attributes. If False, do not capitalize. Default: False.

        Returns
        -------
        list[str]
            Contains the names of the attributes common among modes.
        """
        # get the list of names
        my_name_list = [_k for _k in cls.STELLAR_STRUCTURE.value.keys()] + ['x']
        return [_k.upper() for _k in my_name_list] if upper else my_name_list


# class containing the information stored in the GYRE summary files + unit conversion
class GYRESummaryFiles(BaseFileInfo):
    """Enumeration subclass containing the possible information stored in the GYRE summary files, as well as unit conversion. Inherits the methods from the BaseFileInfo class."""

    # define the elements + their unit conversion methods
    SOLUTION_DATA = {'n_j': {'SI': None}, 'omega': {'SI': None}}
    OBSERVABLES = {
        'freq': {'SI': None},
        'freq_units': {'SI': None},
        'freq_frame': {'SI': None},
        'f_T': {'SI': None},
        'f_g': {'SI': None},
        'psi_T': {'SI': None},
        'psi_g': {'SI': None},
    }
    CLASSIFICATION_VALIDATION = {
        'j': {'SI': None},
        'l': {'SI': None},
        'l_i': {'SI': None},
        'm': {'SI': None},
        'n_p': {'SI': None},
        'n_g': {'SI': None},
        'n_pg': {'SI': None},
        'omega_int': {'SI': None},
    }
    PERTURBATIONS = {
        'x_ref': {'SI': None},
        'xi_r_ref': {'SI': None},
        'eul_phi_ref': {'SI': None},
        'deul_phi_ref': {'SI': None},
        'lag_S_ref': {'SI': None},
        'lag_L_ref': {'SI': None},
    }
    ENERGETICS_TRANSPORT = {
        'eta': {'SI': None},
        'E': {'SI': None},
        'E_p': {'SI': None},
        'E_g': {'SI': None},
        'E_norm': {'SI': None},
        'E_ratio': {'SI': None},
        'H': {'SI': None},
        'W': {'SI': None},
        'tau_ss': {'SI': None},
        'tau_tr': {'SI': None},
    }
    ROTATION = {
        'domega_rot': {'SI': None},
        'dfreq_rot': {'SI': None},
        'beta': {'SI': None},
    }
    STELLAR_STRUCTURE = {
        'M_star': {'SI': None},
        'R_star': {'SI': None},
        'L_star': {'SI': None},
        'Delta_p': {'SI': None},
        'Delta_g': {'SI': None},
    }

    # method used to obtain the mode-independent attributes
    @classmethod
    def retrieve_mode_independent_names(cls) -> list[str]:
        """Method used to retrieve a list of the names of the attributes that are common among modes.

        Returns
        -------
        list[str]
            Contains the mode-independent attribute names.
        """
        return [_k for _k, _ in cls.STELLAR_STRUCTURE.value.items()]


class MESAProfileFiles(BaseFileInfo):
    """Enumeration subclass containing the possible information stored in the MESA profile files, as well as unit conversion. Inherits the methods from the BaseFileInfo class."""

    # custom defined (run_star_extras) isentropic derivatives of Gamma_1
    CUSTOM_THERMODYNAMIC_DERIVATIVES = {'gamma_1_adDer': {'SI': 1e-3}}
    # define the profile elements
    NUCLEAR_ISOTOPES = {
        'h1': {'SI': None},
        'h2': {'SI': None},
        'he3': {'SI': None},
        'he4': {'SI': None},
        'c12': {'SI': None},
        'c13': {'SI': None},
        'n14': {'SI': None},
        'o16': {'SI': None},
        'o17': {'SI': None},
        'ne20': {'SI': None},
        'ne22': {'SI': None},
        'mg24': {'SI': None},
        'al27': {'SI': None},
        'si28': {'SI': None},
        's32': {'SI': None},
        'ar36': {'SI': None},
    }
    CHARACTERISTIC_FREQUENCIES = {
        'brunt_N2': {'SI': None},
        'brunt_N2_structure_term': {'SI': None},
        'brunt_N2_composition_term': {'SI': None},
        'lamb_S': {'SI': None},
        'lamb_S2': {'SI': None},
    }
    HYDROSTATIC_EQUILIBRIUM = {
        'hyeq_lhs': {'SI': None},
        'hyeq_rhs': {'SI': None},
    }
    COMPOSITION = {'x': {'SI': None}, 'y': {'SI': None}, 'z': {'SI': None}}
    MIXING = {
        'mlt_mixing_length': {'SI': 1.0e-2},
        'log_D_mix': {'SI': None},
        'log_D_conv': {'SI': None},
        'log_D_ovr': {'SI': None},
        'log_D_minimum': {'SI': None},
        'conv_mixing_type': {'SI': None},
    }
    GENERIC_PROPERTIES = {
        'zone': {'SI': None},
        'logT': {'SI': None},
        'logRho': {'SI': None},
        'luminosity': {'SI': None},
        'velocity': {'SI': None},
        'entropy': {'SI': None},
        'csound': {'SI': None},
        'mu': {'SI': None},
        'q': {'SI': None},
        'radius': {'SI': None},
        'tau': {'SI': None},
        'pressure': {'SI': 1e-1},
        'opacity': {'SI': None},
        'eps_nuc': {'SI': None},
        'non_nuc_neu': {'SI': None},
        'log_conv_vel': {'SI': None},
        'mass': {'SI': None},
        'mmid': {'SI': None},
    }
    GRADIENTS = {
        'grada': {'SI': None},
        'gradT': {'SI': None},
        'gradr': {'SI': None},
        'pressure_scale_height': {'SI': None},
        'grav': {'SI': 1.0e-2},
    }
    # define the header elements
    CENTER_ISOTOPES = {
        'center_h1': {'SI': None},
        'center_he3': {'SI': None},
        'center_he4': {'SI': None},
        'center_c12': {'SI': None},
        'center_n14': {'SI': None},
        'center_o16': {'SI': None},
        'center_ne20': {'SI': None},
        'center_eta': {'SI': None},
    }
    MASS_ISOTOPES = {
        'star_mass_h1': {'SI': None},
        'star_mass_he3': {'SI': None},
        'star_mass_he4': {'SI': None},
        'star_mass_c12': {'SI': None},
        'star_mass_n14': {'SI': None},
        'star_mass_o16': {'SI': None},
        'star_mass_ne20': {'SI': None},
    }
    CORE_MASSES = {
        'he_core_mass': {'SI': None},
        'c_core_mass': {'SI': None},
        'o_core_mass': {'SI': None},
        'si_core_mass': {'SI': None},
        'fe_core_mass': {'SI': None},
        'neutron_rich_core_mass': {'SI': None},
    }
    TAU_BASED_QUANTITIES = {
        'tau10_mass': {'SI': None},
        'tau10_radius': {'SI': None},
        'tau100_mass': {'SI': None},
        'tau100_radius': {'SI': None},
    }
    INITIAL_PROPERTIES = {
        'initial_z': {'SI': None},
        'initial_mass': {'SI': None},
    }
    PHOTOSPHERIC_QUANTITIES = {
        'photosphere_L': {'SI': None},
        'photosphere_r': {'SI': None},
        'Teff': {'SI': None},
    }
    TIMESCALES = {
        'dynamic_time': {'SI': None},
        'kh_timescale': {'SI': None},
        'nuc_timescale': {'SI': None},
    }
    POWERS = {
        'power_nuc_burn': {'SI': None},
        'power_h_burn': {'SI': None},
        'power_he_burn': {'SI': None},
        'power_neu': {'SI': None},
    }
    GENERIC_ATTRIBUTES = {
        'model_number': {'SI': None},
        'num_zones': {'SI': None},
        'star_age': {'SI': None},
        'time_step': {'SI': None},
        'burn_min1': {'SI': None},
        'burn_min2': {'SI': None},
        'time_seconds': {'SI': None},
        'version_number': {'SI': None},
        'compiler': {'SI': None},
        'build': {'SI': None},
        'MESA_SDK_version': {'SI': None},
        'date': {'SI': None},
    }


# class containing the information stored in the polytrope model files + unit conversion
class PolytropeModelFiles(BaseFileInfo):
    """Enumeration subclass containing the possible information stored in the polytrope model files, as well as unit conversion. Inherits the methods from the BaseFileInfo class."""

    # define the elements + their unit conversion methods
    NUMERICAL_MODEL_PARAMETERS = {'n': {'SI': None}, 'n_r': {'SI': None}}
    STATIC_MODEL_PARAMETERS = {
        'n_poly': {'SI': None},
        'z_b': {'SI': None},
        'Delta_b': {'SI': None},
        'Gamma_1': {'SI': None},
    }
    PROFILES = {
        'z': {'SI': None},
        'theta': {'SI': None},
        'dtheta': {'SI': None},
    }


class PolytropeOscillationModelFiles(BaseFileInfo):
    """Enumeration subclass containing the possible information stored in the polytrope oscillation model files, as well as unit conversion. Inherits the methods from the BaseFileInfo class"""

    # define the elements + their unit conversion methods
    CLASSIFICATION_VALIDATION = {
        'l': {'SI': None},
        'label': {'SI': None},
        'm': {'SI': None},
        'n': {'SI': None},
        'n_g': {'SI': None},
        'n_p': {'SI': None},
        'n_pg': {'SI': None},
        'l_i': {'SI': None},
    }
    OBSERVABLES = {
        'omega': {'SI': None},
        'freq': {'SI': None},
        'freq_frame': {'SI': None},
        'freq_units': {'SI': None},
    }
    PERTURBATIONS = {
        'xi_h': {'SI': None},
        'xi_r': {'SI': None},
        'eul_phi': {'SI': None},
        'deul_phi': {'SI': None},
        'y_1': {'SI': None},
        'y_2': {'SI': None},
        'y_3': {'SI': None},
        'y_4': {'SI': None},
    }
    STELLAR_STRUCTURE = {'Gamma_1': {'SI': None}, 'x': {'SI': None}}
    ROTATION = {'Omega_rot': {'SI': None}, 'lambda': {'SI': None}}

    # method used to obtain the mode-independent attributes
    @classmethod
    def retrieve_mode_independent_names(cls) -> list[str]:
        """Method used to retrieve a list of the names of the attributes that are common among modes.

        Returns
        -------
        list[str]
            The list of mode independent names.
        """
        return ['Omega_rot', 'G', 'M_star', 'R_star', 'Gamma_1']
