"""Python module containing a definition of a class that helps reconstruct a radial kernel,
readying loaded data for plotting purposes.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import collections
import itertools
import numpy as np

# intra-package imports
from .radial_kernel_labeler import generate_plot_labels

# type checking imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # import the radial kernel dict type object
    from ...profile_type_help import RadialKernelDict
    # coupling object
    from ....coupling_coefficients import QCCR


# define a custom object used to load and store radial kernels for plotting purposes
class RadialKernelData:
    """Contains the data necessary to compute the radial kernels for the coupling coefficient expressions."""

    # attribute type declarations
    # - coupler object
    _coupler: 'QCCR'
    # - normalization energy
    _norm_energy: float
    # - stellar evolution model profiles
    gamma_1: np.ndarray
    der_gamma_1_ad: np.ndarray
    pressure: np.ndarray
    density: np.ndarray
    dgravpot: np.ndarray | None
    d2gravpot: np.ndarray | None
    d3gravpot: np.ndarray | None
    radial_coordinate: np.ndarray
    x: np.ndarray
    # - derived profiles
    _gam_min: np.ndarray
    _gam_min_sq: np.ndarray
    _r2: np.ndarray
    _r3: np.ndarray
    # output dictionary
    radial_kernel_terms: 'dict[str, RadialKernelDict]'

    def __init__(self, coupling_object: 'QCCR') -> None:
        # store the coupling object
        self._coupler = coupling_object
        # load stellar evolution model profiles
        self._load_stellar_evolution_model_profiles()
        # load derived profiles
        self._load_derived_profiles()

    def __call__(
        self, contribution_nr: int = 1
    ) -> 'dict[str, RadialKernelDict] | None':
        # initialize a dictionary that will hold the necessary symmetrized radial kernel terms
        self.radial_kernel_terms = {}
        # retrieve the requested necessary radial kernels for a specific contribution to
        # the coupling coefficient and return the result dictionary
        match contribution_nr:
            case 1:
                # load radial kernels for CC contribution 1
                self._load_terms_first_contribution_cc()
            case 2:
                # load radial kernels for CC contribution 2
                self._load_terms_second_contribution_cc()
            case 3:
                # load radial kernels for CC contribution 3
                self._load_terms_third_contribution_cc()
            case 4:
                # load radial kernels for CC contribution 4
                self._load_terms_fourth_contribution_cc()
            case _:
                print(
                    'Unknown case selected. Please select one of the contributi/won numbers 1, 2, 3 or 4 (pointing to one of the four main terms of the coupling coefficient). Will now return "None".'
                )
                return None
        # convert 'legend' strings to plot labels
        generate_plot_labels(radial_kernel_data=self)
        # return the dictionary containing the radial kernels for a specific contribution
        # to the coupling coefficient
        return self.radial_kernel_terms

    # load stellar evolution model profiles
    def _load_stellar_evolution_model_profiles(self) -> None:
        """Load stellar evolution model profiles from the stored mode coupling object."""
        # profiles loaded directly from stellar evolution output
        self._norm_energy = self._coupler._energy_mode_1
        self.gamma_1 = self._coupler._Gamma_1
        self.der_gamma_1_ad = (
            self._coupler._rho * self._coupler._gamma_1_adDer_profile
        )
        self.pressure = self._coupler._P
        self.density = self._coupler._rho
        self.dgravpot = self._coupler._dgravpot
        self.d2gravpot = self._coupler._dgravpot2
        self.d3gravpot = self._coupler._dgravpot3
        self.radial_coordinate = self._coupler._x * self._coupler._R_star
        self.x = self._coupler._x

    # load derived profiles
    def _load_derived_profiles(self) -> None:
        """Loads/stores profiles derived from stellar evolution model profiles."""
        self._gam_min = self.gamma_1 - 1.0
        self._gam_min_sq = self._gam_min**2.0
        self._r2 = self.radial_coordinate**2.0
        self._r3 = self.radial_coordinate**3.0

    # get specific symmetrized radial kernel terms
    def _get_symmetrized_radial_kernel_terms(
        self, kernel_names
    ) -> 'RadialKernelDict':
        """Retrieves symmetrized radial kernel terms based on 'kernel_names'.

        Parameters
        ----------
        kernel_names : list[str]
            Contains a description of 3 specific kernels of which the symmetric product terms
            need to be mapped onto the data of the 3 available modes in the selected triad.

        Returns
        -------
        RadialKernelDict
            Contains the radial kernels and 'legend' strings.
        """
        # get symmetric permutation dict
        _symmetric_perm_dict = self._symmetry_mapping(
            operator_list=kernel_names
        )
        # get radial kernel attributes from the coupler object and the supplied name list;
        # also get the combination string for the plot legend
        my_kernels = np.empty(
            (len(_symmetric_perm_dict), self.x.shape[0]), dtype=np.float64
        )
        plot_legends = []
        for _i, _perm in enumerate(_symmetric_perm_dict.keys()):
            my_kernels[_i, :] = self._kernel_term_constructor(
                radial_terms=_perm
            )
            plot_legends.append(_perm)
        # return the kernels and corresponding legend entries
        return {'kernel': my_kernels, 'legend': plot_legends, 'symmetric': True}

    @staticmethod
    def _symmetry_mapping(
        operator_list: list[str]
    ) -> collections.defaultdict[tuple[str, ...], list]:
        """Internal utility method that generates a mapping of the operators in the 'operator_list'
        to its unique components, uncovering possible symmetries.

        Parameters
        ----------
        operator_list : list[str]
            Describes the operators.

        Returns
        -------
        operator_dict : collections.defaultdict[tuple[str, ...], list]
            Maps unique components to their indices in the symmetrized list.
        """
        # generate the symmetric permutation list
        _sym_trans = list(itertools.permutations(operator_list))
        # use that list to create the mapping dictionary (maps unique components to
        # their indices in the symmetrized list)
        operator_dict = collections.defaultdict(list)
        for _i, _tupe in enumerate(_sym_trans):
            operator_dict[_tupe].append(_i)
        # return the mapping dictionary
        return operator_dict

    def _kernel_term_constructor(
        self, radial_terms: tuple[str, ...]
    ) -> np.ndarray:
        """Internal utility method used to construct the symmetrized term for a radial kernel.

        Parameters
        ----------
        radial_terms : tuple[str, ...]
            Contains the strings that identify different radial terms.

        Returns
        -------
        np.ndarray
            The radial kernel.
        """
        # get the term array
        first_attr = getattr(self._coupler, f'{radial_terms[0]}_mode_1')
        _term_arr = np.empty(
            (first_attr.shape[0], self._coupler._nr_modes), dtype=np.complex128
        )
        _term_arr[:, 0] = first_attr
        for _i, _rt in enumerate(radial_terms[1:]):
            _term_arr[:, _i + 1] = getattr(
                self._coupler, f'{_rt}_mode_{_i + 2}'
            )
        # reconstruct the radial kernel and return it, along with the shape
        return np.prod(_term_arr, axis=1)

    # method used to load a radial kernel term
    def _load_radial_kernel_term(
        self,
        symmetrized_term_list: list[str] | None,
        stellar_evolution_term_list: list[str],
        multiplying_factor: float = 1.0,
        non_symmetrized_term_list: list[str] | tuple[str, ...] | None = None,
    ) -> 'RadialKernelDict':
        """General internal method used to generate and store a radial kernel.

        Parameters
        ----------
        symmetrized_term_list : list[str] | None
            Contains the terms needed to reconstruct the symmetrized radial kernel.
            If None, ignore symmetrization and compute a specific (i.e. non-symmetric) radial kernel,
            if 'non_symmetrized_term_list' is also not None.
        stellar_evolution_term_list : list[str]
            Contains string(s) referring to additional profiles with which
            the radial kernel needs to be multiplied.
        multiplying_factor : float, optional
            Factor with which one needs to multiply the radial kernel; by default 1.0.
        non_symmetrized_term_list : list[str] | None, optional
            Contains the terms needed to reconstruct the specific (non-symmetric) radial kernel,
            if not None; by default None.

        Returns
        -------
        RadialKernelDict
            Contains the necessary data for the radial kernel, such as the 'kernel' values, its plot labels and its multiplication factors.
        """
        if symmetrized_term_list is not None:
            # SYMMETRY
            # load the symmetrized terms
            my_radial_kernel_dict = self._get_symmetrized_radial_kernel_terms(
                kernel_names=symmetrized_term_list
            )
            # specify that this term is symmetrized
            my_radial_kernel_dict['symmetric'] = True
            # perform additional multiplications for the different terms
            my_radial_kernel_dict['kernel'] = (
                my_radial_kernel_dict['kernel']
                * multiplying_factor
                * self._r3
                / self._norm_energy
            )
        else:
            # NO SYMMETRY
            match non_symmetrized_term_list:
                case tuple():
                    rad_tup = non_symmetrized_term_list
                    leg_list = list(non_symmetrized_term_list)
                case list():
                    rad_tup = tuple(non_symmetrized_term_list)
                    leg_list = non_symmetrized_term_list
                case _:
                    raise ValueError(
                        "'non_symmetrized_term_list' should be equal to a list[str] or tuple[str, ...] if 'symmetrized_term_list' is None."
                    )
            # get the radial kernel values and perform multiplication with multiplying factor
            _rad_kernel_values = (
                self._kernel_term_constructor(radial_terms=rad_tup)
                * multiplying_factor
                * self._r3
                / self._norm_energy
            )
            # create the dictionary, specifying that this term is not symmetrized
            my_radial_kernel_dict: 'RadialKernelDict' = {
                'kernel': _rad_kernel_values,
                'legend': leg_list,
                'symmetric': False,
            }
        # perform multiplication with different stellar evolution model profiles
        for _s_e_term in stellar_evolution_term_list:
            # get the stellar evolution profile term
            _s_e_profile = getattr(self, _s_e_term)
            # multiply with the radial kernel terms (relying on automatic broadcasting by numpy)
            my_radial_kernel_dict['kernel'] = (
                my_radial_kernel_dict['kernel'] * _s_e_profile
            )
        # add information about the stellar evolution multipliers to dictionary
        my_radial_kernel_dict['s_e_multipliers'] = stellar_evolution_term_list
        # return the modified radial kernel term dictionary
        return my_radial_kernel_dict

    # load terms of first contribution
    def _load_terms_first_contribution_cc(self) -> None:
        """Store the radial kernels for the first contribution to the coupling coefficient."""
        # store the common list of multiplicative stellar evolution terms
        _contribution_1_s_e_terms = ['pressure', '_gam_min']
        # compute the (symmetrized) different terms and add them to the dictionary
        self.radial_kernel_terms['term_1'] = self._load_radial_kernel_term(
            symmetrized_term_list=[
                '_rad_der_x_z_1',
                '_rad_der_x_z_1',
                '_rad_diverg',
            ],
            stellar_evolution_term_list=_contribution_1_s_e_terms,
            multiplying_factor=0.5,
        )
        self.radial_kernel_terms['term_2'] = self._load_radial_kernel_term(
            symmetrized_term_list=[
                '_z_2_over_c_1_omega',
                '_z_2_over_c_1_omega',
                '_rad_diverg',
            ],
            stellar_evolution_term_list=_contribution_1_s_e_terms,
            multiplying_factor=0.5,
        )
        self.radial_kernel_terms['term_3'] = self._load_radial_kernel_term(
            symmetrized_term_list=['_z_1', '_z_1', '_rad_diverg'],
            stellar_evolution_term_list=_contribution_1_s_e_terms,
            multiplying_factor=1.0,
        )
        self.radial_kernel_terms['term_4'] = self._load_radial_kernel_term(
            symmetrized_term_list=[
                '_z_2_over_c_1_omega',
                '_z_1',
                '_rad_diverg',
            ],
            stellar_evolution_term_list=_contribution_1_s_e_terms,
            multiplying_factor=1.0,
        )
        self.radial_kernel_terms['term_5'] = self._load_radial_kernel_term(
            symmetrized_term_list=[
                '_z_1',
                '_rad_der_x_z_2_c_1_omega',
                '_rad_diverg',
            ],
            stellar_evolution_term_list=_contribution_1_s_e_terms,
            multiplying_factor=1.0,
        )
        self.radial_kernel_terms['term_6'] = self._load_radial_kernel_term(
            symmetrized_term_list=[
                '_rad_diverg',
                '_z_2_over_c_1_omega',
                '_rad_der_x_z_2_c_1_omega',
            ],
            stellar_evolution_term_list=_contribution_1_s_e_terms,
            multiplying_factor=-1.0,
        )  # TODO: CHECK IF THIS TERM WAS CORRECTLY IMPLEMENTED!!!!

    # load terms of second contribution
    def _load_terms_second_contribution_cc(self) -> None:
        """Store the radial kernels for the second contribution to the coupling coefficient."""
        # store the common lists of multiplicative stellar evolution terms
        _contribution_2_s_e_terms_gam_min = ['pressure', '_gam_min_sq']
        _contribution_2_s_e_terms_der_gam = ['pressure', 'der_gamma_1_ad']
        # compute the (non-symmetric) different terms and add them to the dictionary
        self.radial_kernel_terms[
            'term_1_gam_min'
        ] = self._load_radial_kernel_term(
            symmetrized_term_list=None,
            non_symmetrized_term_list=[
                '_rad_diverg',
                '_rad_diverg',
                '_rad_diverg',
            ],
            stellar_evolution_term_list=_contribution_2_s_e_terms_gam_min,
            multiplying_factor=1.0,
        )
        self.radial_kernel_terms[
            'term_1_der_gam'
        ] = self._load_radial_kernel_term(
            symmetrized_term_list=None,
            non_symmetrized_term_list=[
                '_rad_diverg',
                '_rad_diverg',
                '_rad_diverg',
            ],
            stellar_evolution_term_list=_contribution_2_s_e_terms_der_gam,
            multiplying_factor=1.0,
        )  # TODO: CHECK IF NOT SYMMETRIZED IN COMPUTATIONS!

    # load terms of third contribution
    def _load_terms_third_contribution_cc(self) -> None:
        """Store the radial kernels for the third contribution to the coupling coefficient."""
        # store the common list of multiplicative stellar evolution terms
        _contribution_1_s_e_terms = ['pressure']
        # compute the different symmetrized terms and add them to the dictionary
        self.radial_kernel_terms['term_1'] = self._load_radial_kernel_term(
            symmetrized_term_list=[
                '_rad_der_x_z_1',
                '_z_2_over_c_1_omega',
                '_rad_der_x_z_2_c_1_omega',
            ],
            stellar_evolution_term_list=_contribution_1_s_e_terms,
            multiplying_factor=-1.0,
        )
        self.radial_kernel_terms['term_2'] = self._load_radial_kernel_term(
            symmetrized_term_list=[
                '_rad_der_x_z_1',
                '_z_1',
                '_rad_der_x_z_2_c_1_omega',
            ],
            stellar_evolution_term_list=_contribution_1_s_e_terms,
            multiplying_factor=1.0,
        )  # TODO: CHECK IF THIS TERM WAS CORRECTLY IMPLEMENTED!!!!
        self.radial_kernel_terms['term_3'] = self._load_radial_kernel_term(
            symmetrized_term_list=[
                '_z_1',
                '_z_2_over_c_1_omega',
                '_rad_der_x_z_2_c_1_omega',
            ],
            stellar_evolution_term_list=_contribution_1_s_e_terms,
            multiplying_factor=1.0,
        )
        self.radial_kernel_terms['term_4'] = self._load_radial_kernel_term(
            symmetrized_term_list=[
                '_z_2_over_c_1_omega',
                '_z_2_over_c_1_omega',
                '_rad_der_x_z_2_c_1_omega',
            ],
            stellar_evolution_term_list=_contribution_1_s_e_terms,
            multiplying_factor=1.0,
        )
        self.radial_kernel_terms['term_5'] = self._load_radial_kernel_term(
            symmetrized_term_list=['_z_1', '_z_1', '_rad_der_x_z_2_c_1_omega'],
            stellar_evolution_term_list=_contribution_1_s_e_terms,
            multiplying_factor=1.0,
        )
        self.radial_kernel_terms['term_6'] = self._load_radial_kernel_term(
            symmetrized_term_list=[
                '_z_2_over_c_1_omega',
                '_z_2_over_c_1_omega',
                '_z_2_over_c_1_omega',
            ],
            stellar_evolution_term_list=_contribution_1_s_e_terms,
            multiplying_factor=-1.0,
        )
        self.radial_kernel_terms['term_7'] = self._load_radial_kernel_term(
            symmetrized_term_list=[
                '_z_1',
                '_z_2_over_c_1_omega',
                '_z_2_over_c_1_omega',
            ],
            stellar_evolution_term_list=_contribution_1_s_e_terms,
            multiplying_factor=-2.0,
        )
        self.radial_kernel_terms['term_8'] = self._load_radial_kernel_term(
            symmetrized_term_list=[
                '_z_1',
                '_z_2_over_c_1_omega',
                '_z_2_over_c_1_omega',
            ],
            stellar_evolution_term_list=_contribution_1_s_e_terms,
            multiplying_factor=1.0,
        )
        self.radial_kernel_terms['term_9'] = self._load_radial_kernel_term(
            symmetrized_term_list=['_z_1', '_z_1', '_z_2_over_c_1_omega'],
            stellar_evolution_term_list=_contribution_1_s_e_terms,
            multiplying_factor=1.0,
        )
        # compute the non-symmetrized terms
        self.radial_kernel_terms['term_10'] = self._load_radial_kernel_term(
            symmetrized_term_list=None,
            non_symmetrized_term_list=[
                '_rad_der_x_z_1',
                '_rad_der_x_z_1',
                '_rad_der_x_z_1',
            ],
            stellar_evolution_term_list=_contribution_1_s_e_terms,
            multiplying_factor=2.0,
        )
        self.radial_kernel_terms['term_11'] = self._load_radial_kernel_term(
            symmetrized_term_list=None,
            non_symmetrized_term_list=['_z_1', '_z_1', '_z_1'],
            stellar_evolution_term_list=_contribution_1_s_e_terms,
            multiplying_factor=4.0,
        )
        self.radial_kernel_terms['term_12'] = self._load_radial_kernel_term(
            symmetrized_term_list=None,
            non_symmetrized_term_list=[
                '_z_2_over_c_1_omega',
                '_z_2_over_c_1_omega',
                '_z_2_over_c_1_omega',
            ],
            stellar_evolution_term_list=_contribution_1_s_e_terms,
            multiplying_factor=2.0,
        )  # TODO: check if term was correctly implemented!!!

    # load terms of fourth contribution
    def _load_terms_fourth_contribution_cc(self) -> None:
        """Store the radial kernels for the fourth contribution to the coupling coefficient."""
        # store the common lists of multiplicative stellar evolution terms
        _contribution_1_s_e_terms_d3gravpot = ['density', '_r3', 'd3gravpot']
        _contribution_1_s_e_terms_d2gravpot = ['density', '_r2', 'd2gravpot']
        _contribution_1_s_e_terms_dgravpot = [
            'density',
            'radial_coordinate',
            'dgravpot',
        ]
        # compute the different terms and add them to the dictionary
        self.radial_kernel_terms['term_1'] = self._load_radial_kernel_term(
            symmetrized_term_list=[
                '_z_1',
                '_z_2_over_c_1_omega',
                '_z_2_over_c_1_omega',
            ],
            stellar_evolution_term_list=_contribution_1_s_e_terms_dgravpot,
            multiplying_factor=-0.5,
        )
        self.radial_kernel_terms['term_2'] = self._load_radial_kernel_term(
            symmetrized_term_list=[
                '_z_1',
                '_z_2_over_c_1_omega',
                '_z_2_over_c_1_omega',
            ],
            stellar_evolution_term_list=_contribution_1_s_e_terms_d2gravpot,
            multiplying_factor=0.5,
        )
        # compute the non-symmetrized terms
        self.radial_kernel_terms['term_3'] = self._load_radial_kernel_term(
            symmetrized_term_list=None,
            non_symmetrized_term_list=['_z_1', '_z_1', '_z_1'],
            stellar_evolution_term_list=_contribution_1_s_e_terms_d3gravpot,
            multiplying_factor=1.0,
        )
