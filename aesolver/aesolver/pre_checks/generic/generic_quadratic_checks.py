"""Python module containing generic/superclass used to perform pre-checks for quadratic coupling coefficients.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import logging
import sys

# type checking imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


# set up logger
logger = logging.getLogger(__name__)


# create the generic/superclass used to perform prechecks
class PreCheckQuadratic:
    """Class used to perform (generic) pre-checks for quadratic coupling coefficients.

    Notes
    -----
    The only generic quadratic pre-checks are those that rely on the selection rules of the quadratic coupling coefficients, which are implemented/computed/verified using this class.

    Parameters
    ----------
    mode_info_object : list[dict[str, Any]]
        The object containing information on the modes.
    triads : list[tuple[int, int, int]] | None, optional
        Denotes the mode numbers of the modes in the triplet/multiplet; by default None, so that [(1, 2, 3)] becomes the default stored value.
    conjugation : list[tuple[bool, bool, bool]] | None, optional
        Denotes which of the modes are complex-conjugated in the integrals over the volume implied by the inner product of modes, that is, which of the modes :math: `\neq 1` that are complex conjugated (i.e. barred index), and whether mode 1 is barred in the symbol for the coupling coefficient. NOTE: This differs from the frequency handler implementation!
    all_fulfilled : bool, optional
        Determines whether one should check whether all conditions are OK for the generic check (if True), or whether conditions are handled on a case-by-case basis, by default False.
    """

    # attribute type declarations
    _all_fulfilled: bool
    _gyre_ad: 'list[dict[str, Any]]'
    _mode_triads: list
    _conj_triads: list
    _my_range: range
    _m_list: list[list[int]]
    _l_list: list[list[int]]
    _k_list: list[list[int]]

    def __init__(
        self,
        gyre_ad: 'list[dict[str, Any]]',
        triads: list[tuple[int, int, int]] | None = None,
        conjugation: list[tuple[bool, bool, bool] | bool] | None = None,
        all_fulfilled: bool = True,
    ) -> None:
        # default value initialization
        if triads is None:
            triads = [(1, 2, 3)]
        if conjugation is None or isinstance(conjugation[0], bool):
            conjugation = [(True, False, False)]
        # store the boolean that dictates how outputs are generated
        self._all_fulfilled = all_fulfilled
        # store the mode info object
        self._gyre_ad = gyre_ad
        # store the mode triads as a generator
        self._mode_triads = triads
        # store the mode conjugation information
        self._conj_triads = conjugation
        # store the range parameter
        self._my_range = range(len(triads))
        # store relevant information for the checks
        self._store_relevant_info()

    @property
    def l_list(self) -> list[list[int]]:
        """Returns the list of spherical degrees.

        Returns
        -------
        list[int]
            Contains the spherical degrees.
        """
        return self._l_list

    @property
    def m_list(self) -> list[list[int]]:
        """Returns the list of azimuthal orders.

        Returns
        -------
        list[int]
            Contains the azimuthal orders.
        """
        return self._m_list

    @property
    def k_list(self) -> list[list[int]]:
        """Returns the list of meridional degrees.

        Returns
        -------
        list[int]
            Contains the meridional degrees.
        """
        return self._k_list

    # method used to extract and store relevant information for the generic checks
    def _store_relevant_info(self) -> None:
        """Extracts and stores relevant information used to perform the generic checks."""
        # initialize the empty information lists
        self._m_list = []
        self._l_list = []
        self._k_list = []
        # store relevant information for the triads
        for _i, _m in enumerate(self._mode_triads):
            # store the proxy for the spherical degree
            self._l_list.append(self._mode_attr_list_extractor('l', _m))
            # store the azimuthal order
            self._m_list.append(self._mode_attr_list_extractor('m', _m))
            # store the meridional order
            self._k_list.append(self._compute_meridional_orders(_i, _m))

    # attribute list extractor
    def _mode_attr_list_extractor(
        self, my_attr: str, triplet_number_list: list[int]
    ) -> 'list[Any]':
        """Extracts a requested list of mode attributes.

        Parameters
        ----------
        my_attr : str
            Denotes which attribute is requested.
        triplet_number_list : list[int]
            Contains the numbers of the selected modes in the triplet.

        Returns
        -------
        list
            Contains the mode attributes requested.
        """
        try:
            return [
                self._gyre_ad[_g - 1][f'{my_attr}']
                for _g in triplet_number_list
            ]
        except KeyError:
            logger.exception(
                'Wrong key passed to info dictionary. Now exiting.'
            )
            sys.exit()

    # - compute the meridional orders
    def _compute_meridional_orders(
        self, lm_index: int, my_mode_nrs: list[int]
    ) -> list[int]:
        """Computes the meridional order based on the stored spherical degrees and azimuthal orders.

        Parameters
        ----------
        lm_index : int
            The index that covers the different l/m list combinations.
        my_mode_nrs : list[int]
            Contains the mode numbers of the specific mode triplet for which the meridional order is computed.

        Returns
        -------
        list[int]
            Meridional order list.
        """
        # get the respective l and m lists
        _l_list = self._l_list[lm_index]
        _m_list = self._m_list[lm_index]
        # return the meridional order list
        return [
            self._compute_k_grav_in(_l_list[mnr - 1], _m_list[mnr - 1])
            for mnr in my_mode_nrs
        ]

    # utility method used to perform 'm' conjugation
    @staticmethod
    def _conjer_m(conj_var: bool, m: int) -> int:
        """Performs conjugation of the azimuthal order input value, if necessary.

        Parameters
        ----------
        conj_var : bool
            If True, return the conjugate 'm' value. If False, return the 'm' value.
        m : int
            The azimuthal order.

        Returns
        -------
        int
            The conjugate or non-conjugate 'm' value.
        """
        return -m if conj_var else m

    # utility method used to adapt to complex conjugation
    def _adapt_complex_conj_m(
        self, m_tuple: tuple[int, int, int], conj_tuple: tuple[bool]
    ) -> list[int]:
        """Adapts sign(s) of 'm' to the correct one for complex conjugate variables, if necessary.

        Parameters
        ----------
        m_tuple : tuple[int, int, int]
            Contains values for 'm' considered for the current triplet.
        conj_tuple : tuple[bool]
            Defines which of the triplet constituents need to be complex conjugated.

        Returns
        -------
        list[int]
            The list of adapted complex values.
        """
        try:
            return list(map(self._conjer_m, conj_tuple, m_tuple))
        except ValueError:
            logger.exception(
                'Wrong input types passed to this method. Now exiting.'
            )
            sys.exit()

    # utility method used to compute 'k' values for gravito-inertial modes
    @staticmethod
    def _compute_k_grav_in(l_val: int, m_val: int) -> int:
        """Computes the gravito-inertial mode meridional order 'k'.

        Parameters
        ----------
        l_val : int
            The proxy for the spherical degree of a gravito-inertial mode.
        m_val : int
            The azimuthal order of the gravito-inertial mode.

        Returns
        -------
        int
            The meridional order of the gravito-inertial mode.
        """
        return l_val - abs(m_val)

    # individual triplet checking methods
    # - method that enforces the azimuthal selection ('m-selection') rule
    def _azimuthal_selection_rule(self, my_index: int) -> bool:
        """Internal method that implements the azimuthal 'm' selection rule for coupling coefficients in rotating stars.

        Parameters
        ----------
        my_index : int
            The index of the specific mode triplet for which the azimuthal selection rule is enforced.

        Returns
        -------
        bool
            True if the phi integration result is not zero (i.e. enforcing the azimuthal or 'm' selection rule).
        """
        # get the azimuthal wavenumbers
        my_m_tuple = tuple(self._m_list[my_index])
        if TYPE_CHECKING:
            assert len(my_m_tuple) == 3
        # perform conjugation where necessary
        m_conj = self._adapt_complex_conj_m(
            my_m_tuple, self._conj_triads[my_index]
        )
        # return the selection rule result
        return sum(m_conj) == 0

    # - method that enforces the meridional ('k-selection') rule
    def _meridional_selection_rule(self, my_index: int) -> bool:
        """Internal method that implements the meridional 'k' selection rule for quadratic coupling coefficients of gravito-inertial modes in rotating stars.

        Parameters
        ----------
        my_index : int
            The index of the specific mode triplet for which the meridional selection rule is enforced.

        Returns
        -------
        bool
            True if the meridional selection rule holds. False if not.
        """
        # return the selection rule result
        return (sum(self._k_list[my_index]) % 2) == 0

    # looped triplet checking method
    def generic_check(self) -> list[bool] | bool:
        """Checks if the generic pre-check conditions are fulfilled.

        Returns
        -------
        list[bool] or bool
            Depending on the value of the boolean instance attribute '_all_fulfilled' return a list of booleans ('_all_fulfilled'=False), or a boolean ('_all_fulfilled'=True)
        """
        # check if the azimuthal and meridional selection rule per triplet are fulfilled
        az_mer_rules_check = [
            self._azimuthal_selection_rule(_i)
            and self._meridional_selection_rule(_i)
            for _i in self._my_range
        ]
        # check if all the selected triads fulfill the rules, if necessary, otherwise return list of selection results
        if self._all_fulfilled:
            # whether all selected triads fulfill the selection rules
            return all(az_mer_rules_check)
        else:
            # list of selection results
            return az_mer_rules_check
