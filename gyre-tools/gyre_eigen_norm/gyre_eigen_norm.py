"""Python module containing a class that will perform the eigenfunction normalization for the GYRE mode eigenfunctions.

Notes
-----
Requires the user to pass a custom class that lists the eigenfunctions you want to normalize, in addition to a data object that contains a method to compute the mode energy. This was specifically written for a quadratic non-linear mode coupling case in the framework of Van Beeck et al. (2023) but is generalizable to other cases.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import logging
import numpy as np

# local custom enumeration module imports
from .enumeration_modules.enum_norm import EnumNormConvFactors


# set up logger
logger = logging.getLogger(__name__)


# define the class that performs the eigenfunction normalization
class GYREEigenFunctionNormalizer:
    """Class that performs the normalization of the GYRE eigenfunctions (for rotating stars within the TAR) according to several conventions in asteroseismic literature.

    Notes
    -----
    Currently implements the following automatic normalization conventions used by:

    - Lee (2012), in which the (rotating) mode energy is used for normalization.
    - Schenk et al. (2002), in which the classical mode norm is used for normalization.

    Other normalization conventions need to be added by hand. We assume a time dependence of the modes of e^(-i*omega*t), similar to Schenk et al. (2002) and Prat et al. (2019).

    Parameters
    ----------
    norm_convention : str, optional
        The string that denotes the normalization convention. NOTE: Only 'lee_2012' and 'schenk_2002' are supported; by default 'lee_2012'.
    my_obj : QuadraticCouplingCoefficientRotating or None, optional
        The object holding the eigenfunctions; by default None (no object used).
    nr_modes : int, optional
        The number of modes for which the normalization needs to be done. Assumes modes are numbered from 1 up to 'nr_modes'; by default 3.
    iterative_process : bool, optional
        If True and check fails, perform the normalization of the eigenfunction(s) for which the check fails/checks fail iteratively. If False, perform a single-shot renormalization of the eigenfunctions; by default False.
    max_iterations : int, optional
        The maximum number of iterations for the iterative refinement; by default 10.
    """

    # attribute type declarations
    _norm_conv: str
    _data_obj: object
    _coupler_enum: object
    _nr_modes: int
    _norm_factors: np.ndarray
    _check_array: np.ndarray
    _iterative_kwargs: dict
    _iteration_nr: int

    # initialization method
    def __init__(
        self,
        my_obj: object,
        coupler_enumeration: object,
        norm_convention: str = 'lee2012',
        nr_modes: int = 3,
        iterative_process: bool = False,
        max_iterations: int = 10,
    ) -> None:
        # store the normalization convention
        self._norm_conv = norm_convention
        # store the data object (or None, if none is used)
        self._data_obj = my_obj
        # store the coupler enumeration module link
        self._coupler_enum = coupler_enumeration
        # store the number of modes for which normalization needs to be performed
        self._nr_modes = nr_modes
        # store a specific empty array that will hold the normalization factors
        self._norm_factors = np.empty((nr_modes,), dtype=np.float64)
        # store a specific boolean array used to address the need for a potentially iterative renormalization process (if check fails)
        self._check_array = np.ones((nr_modes,), dtype=np.bool_)
        # store the iteration tuple information object
        self._iterative_kwargs = {
            'perform_iteration': iterative_process,
            'nr_max_iterations': max_iterations,
        }
        # store a variable that holds how many normalizations have been done iteratively
        self._iteration_nr = 0

    # call method
    def __call__(self) -> None:
        # perform initial normalization and store initial normalization factors
        self._do_initial_normalization()
        # re-compute the energies (locally or within object)
        self._compute_mode_energies()
        # check initial normalization, and flag candidates for which potentially an additional normalization step is needed
        self._checks_normalizations()
        # perform further iterative re-normalization, if necessary
        self._iteratively_refine()

    # CLASS PROPERTIES

    @property
    def normalization_convention(self):
        """Returns the normalization convention.

        Returns
        -------
        str
            The normalization convention.
        """
        return self._norm_conv

    @property
    def normalizing_factors(self):
        """Returns the normalization factors for the GYRE eigenfunctions of the set of given modes, based on the specific convention selected.

        Returns
        -------
        np.ndarray
            The array of normalization factors.
        """
        return self._norm_factors

    @property
    def normalization_check_array(self):
        """Returns the result of the normalization checks.

        Returns
        -------
        np.ndarray
            The array containing the (boolean) results of the checks.
        """
        return self._check_array

    @property
    def iteration_characteristics(self):
        """Returns the iterative refinement characteristics.

        Returns
        -------
        tuple
            First value in tuple = whether iterative refinement is performed. Second value in tuple = maximum number of iterations. Third value in tuple = number of iterations used during refinement.
        """
        return *self._iterative_kwargs.values(), self._iteration_nr

    # INITIALIZATION HELPER METHODS

    # -- OBJECT-BASED

    # method that performs the initial normalization for the modes
    def _do_initial_normalization(self):
        """Internal utility method that performs the initial normalization of the modes.
        """
        # object mode: retrieve and store the normalization factors based on the information stored within the data object, and then perform initial normalization of mode eigenfunctions
        for _i in range(1, self._nr_modes + 1):
            # get and store the initial normalization factor for mode _i
            self._get_norm_factor(mode_nr=_i)
            # perform the normalization for mode _i
            self._eigenmode_normalizer(mode_nr=_i)

    # method that computes the mode energies for the modes
    def _compute_mode_energies(self):
        """Internal utility method that computes mode energies."""
        # object mode: modes computed within object
        for _i in range(1, self._nr_modes + 1):
            self._data_obj.compute_b_a_mode_energy(_i)  # type: ignore

    # method that performs the checks for the mode normalization factors.
    def _checks_normalizations(self):
        """Internal utility method that performs the check of the mode normalization factors."""
        # object mode: perform checks
        for _i in range(1, self._nr_modes + 1):
            # verify if the check needs to be performed
            if self._check_array[_i - 1]:
                # perform the normalization factor check for mode _i
                self._check_normalization(mode_nr=_i)

    # method that iteratively refines the re-normalization
    def _iteratively_refine(self):
        """Internal utility method that performs the iterative refinement of any of the mode normalization values/factors."""
        # preliminary check if any refinement needs to be done after the first step
        if (
            self._iterative_kwargs['perform_iteration']
            and self._check_array.any()
        ):
            # perform a for loop with maximum number of iterations
            self._iterative_refinement_looper()
        elif self._check_array.any():
            # - get the mode number(s) for modes whose
            # normalization is slightly off
            _refine_mode_nrs = self._check_array.nonzero()[0] + 1
            # compute the percentage deviation from full normalization and issue a warning if necessary
            self._log_debug_if_not_fully_successful(
                refine_mode_nrs=_refine_mode_nrs
            )

    def _iterative_refinement_looper(self):
        """Internal utility method that performs iterative refinement based on iteration parameters."""
        # perform a for loop with maximum number of iterations
        for _ in range(self._iterative_kwargs['nr_max_iterations']):
            # BREAK condition
            if ~self._check_array.any():
                break
            # increase the iteration number by one
            self._iteration_nr += 1
            # get the mode number(s) for which iterative refinement is needed
            _refine_mode_nrs = self._check_array.nonzero()[0] + 1
            # perform refinement
            for _rnr in _refine_mode_nrs:
                self._refine_mode_obj(mode_nr=_rnr)
        else:
            # compute the percentage deviation from full normalization if refinement fails, and issue a warning.
            self._log_debug_if_not_fully_successful(
                refine_mode_nrs=_refine_mode_nrs,  # type: ignore
                optional_additional_message=f'(even after {self._iteration_nr} iterations!).)',
            )

    def _log_debug_if_not_fully_successful(
        self, refine_mode_nrs, optional_additional_message=''
    ):
        """Logs warning if not entirely successful in (iterative) refinement.

        Parameters
        ----------
        refine_mode_nrs : list
            Contains the mode numbers for which percentage deviations of mode properties should be checked.
        optional_additional_message : str, optional
            Optional appended additional message to the warning, if necessary; by default: '' (no additional message).
        """
        # compute the percentage deviation from full normalization
        _proc_devs = self._proc_devs(refine_mode_nrs=refine_mode_nrs)
        # issue warning that normalization is slightly off
        logger.debug(
            f'The normalization for mode(s) {refine_mode_nrs} is slightly off. The percentage deviation(s) from full normalization is(are) {_proc_devs}%. ' + optional_additional_message
        )

    # compute percentage deviation from normalization
    def _proc_devs(self, refine_mode_nrs):
        """Compute the percentage deviations from normalization for the modes.

        Parameters
        ----------
        refine_mode_nrs : np.ndarray
            Contains the number(s) of the mode(s) whose normalization is slightly off.

        Returns
        -------
        proc_devs : np.ndarray
            Contains the percentage deviations from normalization.
        """
        # initialize the difference array
        proc_devs = np.empty((refine_mode_nrs.shape[0],), dtype=np.float64)
        # object mode: perform checks
        for _i, _ref_nr in enumerate(refine_mode_nrs):
            proc_devs[_i] = EnumNormConvFactors.proc_diff_from_object(
                my_obj=self._data_obj,
                normalization_action_string=self._norm_conv,
                my_mode_nr=_ref_nr,
            )
        # return the percentage deviations
        return proc_devs

    # method that performs iterative refining of mode normalization factors
    def _refine_mode_obj(self, mode_nr):
        """Method that refines specific normalization factors for mode eigenfunctions, if necessary.

        Parameters
        ----------
        mode_nr : int
            The number of the specific mode for which the normalization factor needs to be computed.
        """
        # get the new multiplicative normalization factor
        _new_norm_factor = EnumNormConvFactors.factor_from_object(
            my_obj=self._data_obj,
            normalization_action_string=self._norm_conv,
            my_mode_nr=mode_nr,
        )
        # multiply this with the old factor to get the overall normalization factor
        self._norm_factors[mode_nr - 1] *= _new_norm_factor
        # re-normalize the eigenfunctions
        for _attr_base_name in self._coupler_enum.access_all_parameters():  # type: ignore
            # construct the attribute name
            _attr_name = f'{_attr_base_name}_mode_{mode_nr}'
            # perform the normalization, updating the attribute in the data object (NOTE) Need to use new normalization factor because eigenfunctions are already scaled with previous factor!
            setattr(
                self._data_obj,
                _attr_name,
                _new_norm_factor * getattr(self._data_obj, _attr_name),
            )
        # re-compute the mode energies
        # - modes computed within object
        self._data_obj.compute_b_a_mode_energy(mode_nr)  # type: ignore
        # re-do the check of the normalization
        self._check_normalization(mode_nr=mode_nr)

    # method that retrieves the normalization factor from an object for a single mode
    def _get_norm_factor(self, mode_nr):
        """Internal workhorse/utility method that retrieves the normalization factors for individual modes for which info is stored in a specific data object.

        Parameters
        ----------
        mode_nr : int
            The number of the specific mode for which the normalization factor needs to be computed.
        """
        # compute, retrieve and store the normalization factor for the specific mode, based on the data object
        self._norm_factors[
            mode_nr - 1
        ] = EnumNormConvFactors.factor_from_object(
            my_obj=self._data_obj,
            normalization_action_string=self._norm_conv,
            my_mode_nr=mode_nr,
        )

    # method that performs the actual normalization of the eigenfunctions
    def _eigenmode_normalizer(self, mode_nr):
        """Internal workhorse/utility method that performs the normalization of the GYRE mode eigenfunctions for individual modes for which info is stored in a specific data object.

        Parameters
        ----------
        mode_nr : int
            The number of the specific mode for which the normalization factor needs to be computed.
        """
        # loop over the names of all parameters that need to be normalized
        for _attr_base_name in self._coupler_enum.access_all_parameters():  # type: ignore
            # construct the attribute name
            _attr_name = f'{_attr_base_name}_mode_{mode_nr}'
            # perform the normalization, updating the attribute in the data object
            setattr(
                self._data_obj,
                _attr_name,
                self._norm_factors[mode_nr - 1]
                * getattr(self._data_obj, _attr_name),
            )

    # method that checks the normalization factors
    def _check_normalization(self, mode_nr):
        """Internal workhorse/utility method that performs the check for the normalization of the mode eigenfunction, and stores the outcome of that check.

        Parameters
        ----------
        mode_nr : int
            The number of the specific mode for which the normalization check needs to be performed.
        """
        # perform the check of the normalization factor for a specific mode, and store the result of that check in the appropriate place. NOTE: False outcome of the check results in True: result for the check array!
        self._check_array[
            mode_nr - 1
        ] = ~EnumNormConvFactors.check_value_from_object(
            my_obj=self._data_obj,
            normalization_action_string=self._norm_conv,
            my_mode_nr=mode_nr,
        )
