"""Defines a data object class to be used for testing. Mock-up of the actual data object class containing only the methods and attributes that are necessary to test the code in this package!

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import numpy as np

# import custom modules
from num_integrator import cni  # type: ignore


# define the mock-up data object class
class MockDataObj:
    """Mock-up data object class to be used for testing purposes."""

    # attribute type declarations
    _nr_modes: int
    _G: float
    _M_star: float
    _R_star: float
    _b_a_mode_1: float
    _b_a_mode_2: float
    _b_a_mode_3: float
    _energy_mode_1: float
    _energy_mode_2: float
    _energy_mode_3: float
    _classic_norm_mode_1: float
    _classic_norm_mode_2: float
    _classic_norm_mode_3: float
    _z_1_mode_1: np.ndarray
    _z_1_mode_2: np.ndarray
    _z_1_mode_3: np.ndarray
    _z_2_mode_1: np.ndarray
    _z_2_mode_2: np.ndarray
    _z_2_mode_3: np.ndarray
    _x_z_1_mode_1: np.ndarray
    _x_z_1_mode_2: np.ndarray
    _x_z_1_mode_3: np.ndarray
    _z_2_over_c_1_omega_mode_1: np.ndarray
    _z_2_over_c_1_omega_mode_2: np.ndarray
    _z_2_over_c_1_omega_mode_3: np.ndarray
    _x_z_2_over_c_1_omega_mode_1: np.ndarray
    _x_z_2_over_c_1_omega_mode_2: np.ndarray
    _x_z_2_over_c_1_omega_mode_3: np.ndarray
    _rad_der_x_z_1_mode_1: np.ndarray
    _rad_der_x_z_1_mode_2: np.ndarray
    _rad_der_x_z_1_mode_3: np.ndarray
    _rad_der_x_z_2_c_1_omega_mode_1: np.ndarray
    _rad_der_x_z_2_c_1_omega_mode_2: np.ndarray
    _rad_der_x_z_2_c_1_omega_mode_3: np.ndarray
    _rad_diverg_mode_1: np.ndarray
    _rad_diverg_mode_2: np.ndarray
    _rad_diverg_mode_3: np.ndarray
    _lag_L_dim_mode_1: np.ndarray
    _lag_L_dim_mode_2: np.ndarray
    _lag_L_dim_mode_3: np.ndarray
    _hr_mode_1: np.ndarray
    _hr_mode_2: np.ndarray
    _hr_mode_3: np.ndarray
    _ht_mode_1: np.ndarray
    _ht_mode_2: np.ndarray
    _ht_mode_3: np.ndarray
    _hp_mode_1: np.ndarray
    _hp_mode_2: np.ndarray
    _hp_mode_3: np.ndarray
    _norm_rho: np.ndarray
    _x: np.ndarray
    _rho_normed: np.ndarray
    _omp_threads: int
    _use_parallel: bool
    _numerical_integration_method: str
    _surf_rot_freq: float
    _mu_values: np.ndarray
    _corot_mode_omegas: np.ndarray

    def __init__(self, my_input_enum, nr_modes=3, attr_used='_x_z_1') -> None:
        # store the nr. of modes
        self._nr_modes = nr_modes
        # initialize the input data based on an enum
        self._unpack_enum_values(my_enum=my_input_enum)
        # make sure you have owned types of arrays!
        self._owned()
        # use stored enumeration values to generate non-test-dependent values
        self._generate_non_test_dependent_vals(attr_used=attr_used)
        # compute the mode energies and traditional norms
        self._compute_energies_norms()

    def _owned(self):
        """Ensures all enum-read bytes/arrays are writable by making owned copies!"""
        # NECESSARY FOR C++ integration actions!
        self._mu_values = np.array(self._mu_values)
        self._x = np.array(self._x)

    def _unpack_enum_values(self, my_enum):
        """Unpack the stored enumerated input values and keys and store them in this data object.

        Parameters
        ----------
        my_enum : Enum
            The enumeration object holding the input values.
        """
        for _k, _v in my_enum.access_all_input_data(
            nr_modes=self._nr_modes
        ).items():
            setattr(self, _k, _v)

    def _generate_non_test_dependent_vals(self, attr_used='_x_z_1'):
        """Stores values that are necessary for the object but which are not explicitly used in the test method/function.

        Parameters
        ----------
        attr_used : str
            Denotes the attribute used to set the values of the non-test-dependent attributes.
        """
        # loop over all modes to set non-mode-dependent attributes
        for _i in range(1, self._nr_modes + 1):
            # get mode string
            mod_str = f'_mode_{_i}'
            # get value to be used based for the non-mode-dependent attributes
            val_used = getattr(self, f'{attr_used}{mod_str}')
            # set non-test-dependent values
            setattr(self, f'_z_1{mod_str}', val_used)
            setattr(self, f'_z_2{mod_str}', val_used)
            setattr(self, f'_z_2_over_c_1_omega{mod_str}', val_used)
            setattr(self, f'_rad_der_x_z_1{mod_str}', val_used)
            setattr(self, f'_rad_der_x_z_2_c_1_omega{mod_str}', val_used)
            setattr(self, f'_rad_diverg{mod_str}', val_used)
            setattr(self, f'_lag_L_dim{mod_str}', val_used)

    def _compute_traditional_norm_mode(self, mode_nr):
        """Mock-up method of the traditional norm computing method in the actual data object.

        Parameters
        ----------
        mode_nr : int
            Index used to compute the traditional norm of a specific mode.
        """
        # init var
        trad_norm = 0.0
        # init instruction lists
        _h_list = [['hr', 'hr'], ['ht', 'ht'], ['hp', 'hp']]
        _r_list = [
            f'_x_z_1_mode_{mode_nr}',
            f'_x_z_2_over_c_1_omega_mode_{mode_nr}',
            f'_x_z_2_over_c_1_omega_mode_{mode_nr}',
        ]
        # compute common prefactor
        _prefactor = self._x ** (2.0) * self._rho_normed
        # compute integrals
        for _hl, _rl in zip(_h_list, _r_list):
            # compute angular term
            _tn_angular = self._quadratic_angular_integral(
                h_list=_hl, mode_nr=mode_nr, mul_with_mu=False
            )
            # compute radial term integrand
            _rad_integrand = _prefactor * getattr(self, _rl) ** (2.0)
            # compute radial integral
            _tn_radial = self._radint(_rad_integrand, self._x)
            # compute result specific term trad norm
            trad_norm += _tn_angular * _tn_radial
        # multiply with prefactor (due to Phi coordinate integration) to get traditional norm
        setattr(self, f'_classic_norm_mode_{mode_nr}', trad_norm * 2.0 * np.pi)

    def _quadratic_angular_integral(self, h_list, mode_nr, mul_with_mu=False):
        """Mock-up of the quadratic angular integral method used in the actual data object.

        Parameters
        ----------
        h_list : list[str]
            Contains the Hough function/derivative descriptors.
        mode_nr : int
            Index used to compute angular integral for a specific mode.
        mul_with_mu : bool, optional
            If True, multiplies the integrand with the mu values before integration; by default False.

        Returns
        -------
        float
            The angular integration result.
        """
        # get the descriptors
        descriptors = self._map_hough_to_mode(h_list, [mode_nr] * 2)
        # get the appropriate size of the internal utility lists
        _app_size = len(descriptors)
        # map descriptions to Hough functions/derivatives
        _first = getattr(self, descriptors[0])
        _map_arr = np.empty((_app_size, _first.shape[0]), dtype=np.float64)
        _map_arr[0, :] = _first
        for _i, _desc in enumerate(descriptors[1:]):
            _map_arr[_i + 1, :] = getattr(self, _desc)
        # compute integrand
        _integrand = np.prod(_map_arr, axis=0)
        # get mu array
        _mu_arr = getattr(self, '_mu_values')
        # check if we need to multiply with these values in the integrand
        if mul_with_mu:
            _integrand *= _mu_arr
        # compute the integral and return result
        return cni.integrate(
            self._numerical_integration_method,
            True,
            _integrand,
            _mu_arr,
            self._omp_threads,
            self._use_parallel,
        )

    def _compute_energies_norms(self):
        """Method that computes the energies and traditional norms of our mock-up data object, initializing the relevant attributes."""
        for _i in range(1, self._nr_modes + 1):
            self.compute_b_a_mode_energy(mode_nr=_i)

    def compute_b_a_mode_energy(self, mode_nr):
        """Mock-up method of the mode energy computing method in the actual data object.

        Parameters
        ----------
        mode_nr : int
            Index used to compute energy of a specific mode.
        """
        # compute the radial pre-factor --> go to dimensioned energy
        _dimension_radial_prefactor = self._norm_rho * self._R_star ** (5.0)
        # compute the traditional norm of the mode
        self._compute_traditional_norm_mode(mode_nr=mode_nr)
        # compute the angular integral
        _coriolis_norm_angular = self._quadratic_angular_integral(
            h_list=['ht', 'hp'], mode_nr=mode_nr, mul_with_mu=True
        )
        # compute the radial integral
        _coriolis_norm_radial_integrand = (
            self._rho_normed
            * (self._x**2.0)
            * getattr(self, f'_x_z_2_over_c_1_omega_mode_{mode_nr}') ** 2.0
        )
        _coriolis_norm_radial = self._radint(
            _coriolis_norm_radial_integrand, self._x
        )
        # compute the Coriolis contribution
        _coriolis_norm = (
            _coriolis_norm_radial
            * _coriolis_norm_angular
            * 8.0
            * np.pi
            * self._surf_rot_freq
        )
        # get corotating-frame frequency for the mode
        my_corot = self._corot_mode_omegas[mode_nr - 1]
        # compute and store the energy normalization factor b_a
        b_a = (
            _coriolis_norm
            + (2.0 * my_corot * getattr(self, f'_classic_norm_mode_{mode_nr}'))
        ) * _dimension_radial_prefactor
        setattr(self, f'_b_a_mode_{mode_nr}', b_a)
        # compute and store oscillation mode energy
        setattr(self, f'_energy_mode_{mode_nr}', my_corot * b_a)

    def _map_hough_to_mode(self, my_list, mode_nrs):
        """Mock-up method of the string mapping method in the actual data object.

        Parameters
        ----------
        my_list : list[str]
            The Hough function (or derivative) descriptors.
        mode_nrs : list[int]
            The indices used to map specific modes to Hough functions.

        Returns
        -------
        list[str]
            List of mapping strings.
        """
        return [f'_{h}_mode_{i}' for h, i in zip(my_list, mode_nrs)]

    def _radint(self, integrand, integration_variable):
        """Mock-up method of the radial integral computing method in the actual data object.

        Parameters
        ----------
        integrand : np.ndarray
            The integrand for the radial integral.
        integration_variable : np.ndarray
            The integration variable for the radial integral.

        Returns
        -------
        float
            The radial integration result.
        """
        return cni.integrate(
            self._numerical_integration_method,
            False,
            np.ascontiguousarray(integrand.real),
            integration_variable,
            self._omp_threads,
            self._use_parallel,
        )
