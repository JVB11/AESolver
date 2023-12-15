"""Python module containing a definition of a class that helps reconstruct an angular kernel, readying loaded data for plotting purposes.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import collections
import itertools
import numpy as np
import logging

# intra-package imports
from .angular_kernel_labeler import generate_angular_plot_labels

# type checking imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # import coupling coefficient object
    from ....coupling_coefficients import QCCR
    # import the angular kernel dict type object
    from ...profile_type_help import AngularKernelDict


logger = logging.getLogger(__name__)


# define a custom object used to load and store angular kernels for plotting purposes
class AngularKernelData:
    """Contains the data necessary to compute the angular kernels for the coupling coefficient expressions."""

    # attribute type declarations
    # - coupler object
    _coupler: 'QCCR'
    # - angular mode-independent profiles
    _mu_vals: np.ndarray
    _div_by_sin_scaling: np.ndarray
    # output dictionary
    angular_kernel_terms: 'dict[str, AngularKernelDict]'

    def __init__(self, coupling_object: 'QCCR') -> None:
        # store the coupling object
        self._coupler = coupling_object
        # load mode-independent angular profiles
        self._load_mode_independent_angular_profiles()

    def __call__(
        self, contribution_nr: int = 1
    ) -> 'dict[str, AngularKernelDict] | None':
        # initialize a dictionary that will hold the necessary symmetrized angular kernel terms
        self.angular_kernel_terms = {}
        # retrieve the requested necessary angular kernels for a specific contribution to the coupling coefficient and return the result dictionary TODO: implement this!
        match contribution_nr:
            case 1:
                # load angular kernels for CC contribution 1
                self._load_terms_first_contribution_cc()
            case 2:
                # load angular kernels for CC contribution 2
                self._load_terms_second_contribution_cc()
            case 3:
                # load angular kernels for CC contribution 3
                self._load_terms_third_contribution_cc()
            case 4:
                # load angular kernels for CC contribution 4
                self._load_terms_fourth_contribution_cc()
            case _:
                logger.error(
                    'Unknown case selected. Please select one of the contribution numbers 1, 2, 3 or 4 (pointing to one of the four main terms of the coupling coefficient). Will now return "None".'
                )
                return None
        # use function to convert 'legend' strings to plot labels
        generate_angular_plot_labels(angular_kernel_data=self)
        # return the dictionary containing the angular kernels for a specific contribution to the coupling coefficient
        return self.angular_kernel_terms

    # load angular profiles
    def _load_mode_independent_angular_profiles(self) -> None:
        """Loads mode-independent angular data profiles from the stored mode coupling object."""
        # get mu values
        self._mu_vals = getattr(self._coupler, '_mu_values')
        # get scaling values for terms divided by sines
        self._div_by_sin_scaling = 1.0 / np.sqrt(1.0 - self._mu_vals**2.0)

    # get specific symmetrized angular kernel terms
    def _get_symmetrized_angular_kernel_terms(
        self, kernel_names
    ) -> 'AngularKernelDict':
        """Retrieves symmetrized angular kernel terms based on 'kernel_names'.

        Parameters
        ----------
        kernel_names : list[str]
            Contains a description of 3 specific kernels of which the symmetric product terms need to be mapped
            onto the data of the 3 available modes in the selected triad.

        Returns
        -------
        dict[str, np.ndarray, list[str] | bool]
            Contains the angular kernels and 'plot_legend' strings.
        """
        # get symmetric permutation dict
        _symmetric_perm_dict = self._symmetry_mapping(
            operator_list=kernel_names
        )
        # get angular kernel attributes from the coupler object and the supplied name list;
        # also get the combination string for the plot legend
        my_kernels = np.empty(
            (len(_symmetric_perm_dict), self._mu_vals.shape[0]),
            dtype=np.float64,
        )
        plot_legends = []
        for _i, _perm in enumerate(_symmetric_perm_dict.keys()):
            my_kernels[_i, :] = self._kernel_term_constructor(
                angular_terms=_perm
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
        operator_dict : defaultdict[tuple[str, ...], list]
            Maps unique components to their indices in the symmetrized list.
        """
        # generate the symmetric permutation list
        _sym_trans = list(itertools.permutations(operator_list))
        # use that list to create the mapping dictionar
        # (maps unique components to their indices in the symmetrized list)
        operator_dict = collections.defaultdict(list)
        for _i, _tupe in enumerate(_sym_trans):
            operator_dict[_tupe].append(_i)
        # return the mapping dictionary
        return operator_dict

    def _kernel_term_constructor(
        self, angular_terms: tuple[str, ...]
    ) -> np.ndarray:
        """Internal utility method used to construct the symmetrized term for an angular kernel.

        Parameters
        ----------
        angular_terms : tuple[str, ...]
            Contains the strings that identify different angular terms.

        Returns
        -------
        np.ndarray
            The angular kernel.
        """
        # get the term array
        first_attr = getattr(self._coupler, f'_{angular_terms[0]}_mode_1')
        _term_arr = np.empty(
            (first_attr.shape[0], self._coupler._nr_modes), dtype=np.complex128
        )
        _term_arr[:, 0] = first_attr
        for _i, _at in enumerate(angular_terms[1:]):
            _term_arr[:, _i + 1] = getattr(
                self._coupler, f'_{_at}_mode_{_i + 2}'
            )
        # reconstruct the angular kernel and return it, along with the shape
        return np.prod(_term_arr, axis=1)

    def _scale_angular_terms(
        self,
        my_kernel: np.ndarray,
        mul_with_mu: bool,
        div_by_sin: bool,
        multiplying_factor: float,
    ) -> None:
        """Scales angular kernels (in-place) with specific factors to obtain further differentiation between terms.

        Parameters
        ----------
        my_kernel : np.ndarray
            Computed product kernel of three different angular mode dependence functions.
        mul_with_mu : bool
            If True, multiply the product kernel with cosine(theta) = Mu.
        div_by_sin : bool
            If True, divide the product kernel by sine(theta).
        multiplying_factor : float
            Multiply the product kernel with this factor. Set to 1.0 if no multiplication is necessary.
        """
        match (mul_with_mu, div_by_sin):
            case (False, False):
                my_kernel = my_kernel * multiplying_factor
            case (True, False):
                my_kernel = my_kernel * multiplying_factor * self._mu_vals
            case (False, True):
                my_kernel = (
                    my_kernel * multiplying_factor * self._div_by_sin_scaling
                )
            case (True, True):
                my_kernel = (
                    my_kernel
                    * multiplying_factor
                    * self._mu_vals
                    * self._div_by_sin_scaling
                )
            case _:
                raise ValueError(f"'mul_with_mu' and 'div_by_sin' should be boolean values, and they were not ('mul_with_mu': {mul_with_mu}, 'div_by_sin': {div_by_sin}).")

    def _load_angular_kernel_term(
        self,
        symmetrized_term_list: list[str] | None,
        multiplying_factor: float = 1.0,
        non_symmetrized_term_list: list[str] | tuple[str, ...] | None = None,
        mul_with_mu: bool = False,
        div_by_sin: bool = False,
    ) -> 'AngularKernelDict':
        """Generic internal method used to load an angular kernel term.

        Parameters
        ----------
        symmetrized_term_list : list[str] | None
            Contains the terms needed to reconstruct the symmetrized angular kernel. If None, ignore symmetrization and compute a specific (i.e. non-symmetric) angular kernel, if 'non_symmetrized_term_list' is not None.
        multiplying_factor : float, optional
            Factor with which one needs to multiply the angular kernel.
            Default Value: 1.0.
        non_symmetrized_term_list : tuple[str, ...] | None, optional
            Contains the terms needed to reconstruct the specific (non-symmetric) angular kernel, if not None.
            Default Value: None.
        mul_with_mu : bool, optional
            If True, multiply the angular kernel with cos(theta) = mu.
            Default Value: False.
        div_by_sin : bool, optional
            If True, divide the angular kernel with sin(theta).
            Default Value: False.

        Returns
        -------
        dict[str, np.ndarray | list[str] | bool | float]
            Contains the necessary data for the angular kernel, such as the 'kernel' values, its plot labels and its multiplication factors.
        """
        if symmetrized_term_list is not None:
            # SYMMETRY
            # load the symmetrized terms
            my_angular_kernel_dict = self._get_symmetrized_angular_kernel_terms(
                kernel_names=symmetrized_term_list
            )
        else:
            # NO SYMMETRY
            match non_symmetrized_term_list:
                case tuple():
                    ang_tup = non_symmetrized_term_list
                    leg_list = list(non_symmetrized_term_list)
                case list():
                    ang_tup = tuple(non_symmetrized_term_list)
                    leg_list = non_symmetrized_term_list
                case _:
                    raise ValueError(
                    "'non_symmetrized_term_list' should be equal to a list[str] or tuple[str, ...] if 'symmetrized_term_list' is None."
                    )
            # get the angular kernel values and perform necessary multiplications
            _ang_kernel_values = self._kernel_term_constructor(
                angular_terms=ang_tup
            )
            # create the dictionary, specifying that this term is not symmetrized
            my_angular_kernel_dict: 'AngularKernelDict' = {
                'kernel': _ang_kernel_values,
                'legend': leg_list,
                'symmetric': False,
            }
        # perform additional scaling multiplications for the different terms
        my_kernel = my_angular_kernel_dict['kernel']
        if TYPE_CHECKING:
            assert isinstance(my_kernel, np.ndarray)
        self._scale_angular_terms(
            my_kernel=my_kernel,
            div_by_sin=div_by_sin,
            mul_with_mu=mul_with_mu,
            multiplying_factor=multiplying_factor,
        )
        # add information about the multipliers to the dictionary
        my_angular_kernel_dict['mul_with_mu'] = mul_with_mu
        my_angular_kernel_dict['div_by_sin'] = div_by_sin
        my_angular_kernel_dict['multiplying_factor'] = multiplying_factor
        # return the modified angular kernel dict
        return my_angular_kernel_dict

    # load terms of first contribution
    def _load_terms_first_contribution_cc(self) -> None:
        """Loads the angular kernels for the first contribution to the coupling coefficient."""
        # load the symmetrized terms related to the first contribution
        self.angular_kernel_terms['term_1'] = self._load_angular_kernel_term(
            symmetrized_term_list=['hr', 'hr', 'hr'],
            mul_with_mu=False,
            div_by_sin=False,
        )
        self.angular_kernel_terms['term_2'] = self._load_angular_kernel_term(
            symmetrized_term_list=[
                'theta_der_ht_theta',
                'theta_der_ht_theta',
                'hr',
            ],
            mul_with_mu=False,
            div_by_sin=False,
        )
        self.angular_kernel_terms['term_3'] = self._load_angular_kernel_term(
            symmetrized_term_list=['phi_phi', 'phi_phi', 'hr'],
            mul_with_mu=False,
            div_by_sin=False,
        )
        self.angular_kernel_terms['term_4'] = self._load_angular_kernel_term(
            symmetrized_term_list=['theta_phi', 'theta_der_hp_theta', 'hr'],
            mul_with_mu=False,
            div_by_sin=True,
            multiplying_factor=-2.0,
        )
        self.angular_kernel_terms['term_5'] = self._load_angular_kernel_term(
            symmetrized_term_list=['theta_der_ht_theta', 'hr', 'hr'],
            mul_with_mu=False,
            div_by_sin=False,
        )
        self.angular_kernel_terms['term_6'] = self._load_angular_kernel_term(
            symmetrized_term_list=['phi_phi', 'hr', 'hr'],
            mul_with_mu=False,
            div_by_sin=False,
        )
        self.angular_kernel_terms['term_7'] = self._load_angular_kernel_term(
            symmetrized_term_list=['theta_der_hr_theta', 'ht', 'hr'],
            mul_with_mu=False,
            div_by_sin=False,
        )
        self.angular_kernel_terms['term_8'] = self._load_angular_kernel_term(
            symmetrized_term_list=['mhr', 'hp', 'hr'],
            mul_with_mu=False,
            div_by_sin=True,
            multiplying_factor=-1.0,
        )
        self.angular_kernel_terms['term_9'] = self._load_angular_kernel_term(
            symmetrized_term_list=['hr', 'ht', 'ht'],
            mul_with_mu=False,
            div_by_sin=False,
        )
        self.angular_kernel_terms['term_10'] = self._load_angular_kernel_term(
            symmetrized_term_list=['hr', 'hp', 'hp'],
            mul_with_mu=False,
            div_by_sin=False,
            multiplying_factor=-1.0,
        )

    # load terms of second contribution
    def _load_terms_second_contribution_cc(self) -> None:
        """Loads angular kernels for the second contribution to the coupling coefficient."""
        # load the non-symmetric terms related to the second contribution
        self.angular_kernel_terms['term_1'] = self._load_angular_kernel_term(
            symmetrized_term_list=None,
            non_symmetrized_term_list=['hr', 'hr', 'hr'],
            mul_with_mu=False,
            div_by_sin=False,
        )

    # load terms of third contribution
    def _load_terms_third_contribution_cc(self) -> None:
        """Loads angular kernels for the third contribution to the coupling coefficient."""
        # load the symmetrized terms related to the third contribution
        self.angular_kernel_terms['term_1'] = self._load_angular_kernel_term(
            symmetrized_term_list=['hr', 'ht', 'ht'],
            mul_with_mu=False,
            div_by_sin=False,
        )
        self.angular_kernel_terms['term_2'] = self._load_angular_kernel_term(
            symmetrized_term_list=['hr', 'hp', 'hp'],
            mul_with_mu=False,
            div_by_sin=False,
            multiplying_factor=-1.0,
        )
        self.angular_kernel_terms['term_3'] = self._load_angular_kernel_term(
            symmetrized_term_list=['theta_der_hr_theta', 'hr', 'ht'],
            mul_with_mu=False,
            div_by_sin=False,
        )
        self.angular_kernel_terms['term_4'] = self._load_angular_kernel_term(
            symmetrized_term_list=['mhr', 'hr', 'hp'],
            mul_with_mu=False,
            div_by_sin=True,
            multiplying_factor=-1.0,
        )
        self.angular_kernel_terms['term_5'] = self._load_angular_kernel_term(
            symmetrized_term_list=[
                'theta_der_hr_theta',
                'theta_der_ht_theta',
                'ht',
            ],
            mul_with_mu=False,
            div_by_sin=False,
        )
        self.angular_kernel_terms['term_6'] = self._load_angular_kernel_term(
            symmetrized_term_list=['mhr', 'phi_phi', 'hp'],
            mul_with_mu=False,
            div_by_sin=True,
        )
        self.angular_kernel_terms['term_7'] = self._load_angular_kernel_term(
            symmetrized_term_list=['mhr', 'theta_der_hp_theta', 'ht'],
            mul_with_mu=False,
            div_by_sin=True,
        )
        self.angular_kernel_terms['term_8'] = self._load_angular_kernel_term(
            symmetrized_term_list=['theta_der_hr_theta', 'theta_phi', 'hp'],
            mul_with_mu=False,
            div_by_sin=True,
        )
        self.angular_kernel_terms['term_9'] = self._load_angular_kernel_term(
            symmetrized_term_list=['ht', 'theta_der_ht_theta', 'ht'],
            mul_with_mu=False,
            div_by_sin=False,
        )
        self.angular_kernel_terms['term_10'] = self._load_angular_kernel_term(
            symmetrized_term_list=['hp', 'phi_phi', 'hp'],
            mul_with_mu=False,
            div_by_sin=False,
        )
        self.angular_kernel_terms['term_11'] = self._load_angular_kernel_term(
            symmetrized_term_list=['hp', 'theta_der_hp_theta', 'ht'],
            mul_with_mu=False,
            div_by_sin=False,
        )
        self.angular_kernel_terms['term_12'] = self._load_angular_kernel_term(
            symmetrized_term_list=['theta_phi', 'ht', 'hp'],
            mul_with_mu=False,
            div_by_sin=True,
        )
        self.angular_kernel_terms['term_13'] = self._load_angular_kernel_term(
            symmetrized_term_list=['theta_der_hr_theta', 'hr', 'ht'],
            mul_with_mu=False,
            div_by_sin=False,
        )
        self.angular_kernel_terms['term_14'] = self._load_angular_kernel_term(
            symmetrized_term_list=['mhr', 'hr', 'hp'],
            mul_with_mu=False,
            div_by_sin=True,
            multiplying_factor=-1.0,
        )
        self.angular_kernel_terms['term_15'] = self._load_angular_kernel_term(
            symmetrized_term_list=[
                'theta_der_ht_theta',
                'theta_phi',
                'theta_der_hp_theta',
            ],
            mul_with_mu=False,
            div_by_sin=True,
        )
        self.angular_kernel_terms['term_16'] = self._load_angular_kernel_term(
            symmetrized_term_list=[
                'theta_der_hp_theta',
                'theta_phi',
                'phi_phi',
            ],
            mul_with_mu=False,
            div_by_sin=True,
        )
        self.angular_kernel_terms['term_17'] = self._load_angular_kernel_term(
            symmetrized_term_list=['hr', 'theta_phi', 'theta_der_hp_theta'],
            mul_with_mu=False,
            div_by_sin=True,
            multiplying_factor=-1.0,
        )
        self.angular_kernel_terms['term_18'] = self._load_angular_kernel_term(
            symmetrized_term_list=['hr', 'phi_phi', 'phi_phi'],
            mul_with_mu=False,
            div_by_sin=True,
        )
        self.angular_kernel_terms['term_19'] = self._load_angular_kernel_term(
            symmetrized_term_list=[
                'hr',
                'theta_der_ht_theta',
                'theta_der_ht_theta',
            ],
            mul_with_mu=False,
            div_by_sin=True,
        )
        self.angular_kernel_terms['term_20'] = self._load_angular_kernel_term(
            symmetrized_term_list=['hr', 'hr', 'theta_der_ht_theta'],
            mul_with_mu=False,
            div_by_sin=False,
        )
        self.angular_kernel_terms['term_21'] = self._load_angular_kernel_term(
            symmetrized_term_list=['hr', 'hr', 'phi_phi'],
            mul_with_mu=False,
            div_by_sin=False,
        )
        # load non-symmetric angular kernels
        self.angular_kernel_terms['term_22'] = self._load_angular_kernel_term(
            symmetrized_term_list=None,
            non_symmetrized_term_list=['hr', 'hr', 'hr'],
            mul_with_mu=False,
            div_by_sin=False,
        )
        self.angular_kernel_terms['term_23'] = self._load_angular_kernel_term(
            symmetrized_term_list=None,
            non_symmetrized_term_list=['phi_phi', 'phi_phi', 'phi_phi'],
            mul_with_mu=False,
            div_by_sin=False,
        )
        self.angular_kernel_terms['term_24'] = self._load_angular_kernel_term(
            symmetrized_term_list=None,
            non_symmetrized_term_list=[
                'theta_der_ht_theta',
                'theta_der_ht_theta',
                'theta_der_ht_theta',
            ],
            mul_with_mu=False,
            div_by_sin=False,
        )

    # load terms of fourth contribution
    def _load_terms_fourth_contribution_cc(self) -> None:
        """Loads angular kernels for the fourth contribution to the coupling coefficient."""
        # load the symmetrized terms related to the fourth contribution
        self.angular_kernel_terms['term_1'] = self._load_angular_kernel_term(
            symmetrized_term_list=['hr', 'ht', 'ht'],
            mul_with_mu=False,
            div_by_sin=False,
        )
        self.angular_kernel_terms['term_2'] = self._load_angular_kernel_term(
            symmetrized_term_list=['hr', 'hp', 'hp'],
            mul_with_mu=False,
            div_by_sin=False,
            multiplying_factor=-1.0,
        )
        # load the non-symmetric terms related to the fourth contribution
        self.angular_kernel_terms['term_3'] = self._load_angular_kernel_term(
            symmetrized_term_list=None,
            non_symmetrized_term_list=['hr', 'hr', 'hr'],
            mul_with_mu=False,
            div_by_sin=False,
        )
