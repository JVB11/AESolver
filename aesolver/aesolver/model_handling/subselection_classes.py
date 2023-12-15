"""File containing python helper classes used to sub-select between files.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
from collections.abc import Callable
import re
from functools import singledispatchmethod

# type checking imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Pattern


# superclass used to define generic methods to sub-select between files
class SubSelectorFiles:
    """Python superclass containing several generic methods used to sub-select between different types of files.

    Parameters
    ----------
    search_elements : dict
        Contains the information necessary to subselect files.
    pattern_dictionary : dict
        Maps to patterns used to subselect files.
    """

    # attribute type declaration
    _search_elements: dict[str, tuple]
    _pattern_dictionary: "dict[str, Pattern]"

    # initialization method
    def __init__(
        self,
        search_elements: dict[str, tuple],
        pattern_dictionary: "dict[str, Pattern[str]]",
    ) -> None:
        # construct the search elements dictionary
        self._search_elements = search_elements
        # store the dictionary that holds regex patterns for conditional checks
        self._pattern_dictionary = pattern_dictionary

    # public, callable method used to assess conditions for sub-selection
    def subselect(
        self,
        name: str,
        additional_substring: list[str] | tuple[str, ...] | str | None = None,
    ) -> bool:
        """Method used to sub-select between different files.

        Parameters
        ----------
        name : str
            The name of the file to be used for sub-selection.
        additional_substring : list[str] or str or None, optional
            An additional substring used for additional sub-selection. This substring needs to be present in the file name; by default None.

        Returns
        -------
        bool
            The sub-selection bool.
        """
        return self._verify_conditions(additional_substring, name)

    # method used to verify you verify one of the conditions for the three mode files
    @singledispatchmethod
    def _verify_conditions(
        self,
        additional_substrings: list[str] | tuple[str, ...] | str | None,
        name: str,
    ) -> bool:
        """Internal dispatched utility method used to match one of the three conditions specified for the mode files.

        Parameters
        ----------
        additional_substrings : : list[str] | tuple[str, ...] | str | None
            An additional substring used for additional sub-selection. (Needs to be present in the file name.)
        name : str
            The name of the file to be used for sub-selection.

        Returns
        -------
        bool
            The sub-selection bool.
        """
        raise NotImplementedError(
            "Could not determine the type of 'additional_substrings'."
        )

    @_verify_conditions.register
    def _(self, additional_substrings: None, name: str) -> bool:
        # loop over the elements of the search dictionary
        for _key_search, _condition_element in self._search_elements.items():
            # break early if one of the statements is not fulfilled
            if not self._conditional(_condition_element, _key_search, name):
                return False
        else:
            # no breaks: select this file whose name meets the conditions
            return True

    @_verify_conditions.register
    def _(self, additional_substrings: list, name: str) -> bool:
        # check the additional substrings
        for _a_s in additional_substrings:
            if _a_s not in name:
                return False
        # loop over the elements of the search dictionary
        for _key_search, _condition_element in self._search_elements.items():
            # break early if one of the statements is not fulfilled
            if not self._conditional(_condition_element, _key_search, name):
                return False
        else:
            # no breaks: select this file whose name meets the conditions
            return True

    @_verify_conditions.register
    def _(self, additional_substrings: tuple, name: str) -> bool:
        # check the additional substrings
        for _a_s in additional_substrings:
            if _a_s not in name:
                return False
        # loop over the elements of the search dictionary
        for _key_search, _condition_element in self._search_elements.items():
            # break early if one of the statements is not fulfilled
            if not self._conditional(_condition_element, _key_search, name):
                return False
        else:
            # no breaks: select this file whose name meets the conditions
            return True

    # internal utility method that verifies conditions for sub-selection
    def _conditional(
        self, conditional_element: list | tuple, element_key: str, name: str
    ) -> bool:
        """Dispatched internal method used for conditional sub-selection of files.

        Parameters
        ----------
        conditional_element : list or tuple
            The condition used to sub-select files.
        element_key: str
            The keyword of the conditional element.
        name : str
            The name of the file that is subjected to conditional tests used to sub-select files.

        Return
        ------
        bool
            Subselection result.
        """
        # retrieve the pattern, and detect if the conditional element
        # is fulfilled
        if _match := self._pattern_dictionary[element_key].match(name):
            if conditional_element[1] and (
                conditional_element[0] == _match.group(1)
            ):
                # positive matching worked: True
                return True
            else:
                # matching a different suffix: False
                return False
        elif not conditional_element[1]:
            # negative matching worked: True
            return True
        else:
            # matching did not work: False
            return False


# subclass used to sub-select between MESA profile files
class MESAProfileSubSelector(SubSelectorFiles):
    """Python class that holds methods used to sub-select between MESA profile files.

    Parameters
    ----------
    mesa_file_suffix : str, optional
        The suffix of the MESA profile files. Default: 'dat'.
    """

    # initialization method
    def __init__(self, mesa_file_suffix: str = 'dat') -> None:
        # initialize the superclass
        super().__init__(
            search_elements={
                'file_suffix': (mesa_file_suffix, True),
                'not_gyre': ('gyre', False),
            },
            pattern_dictionary={
                'file_suffix': re.compile(rf'.+\.({mesa_file_suffix})'),
                'not_gyre': re.compile(r'.+(gyre|GYRE).*'),
            },
        )


# class used to sub-select between GYRE summary files
class GYRESummarySubSelector:
    """Python class that holds methods used to sub-select between
    GYRE summary files.
    """


# class used to sub-select between GYRE detail files
class GYREDetailSubSelector(SubSelectorFiles):
    """Python class that holds methods used to sub-select between
    GYRE detail files.

    Parameters
    ----------
    search_l : list[int], optional
        The list of spherical degrees used for searching for GYRE detail files; by default [1,1,1].
    search_m : list[int], optional
        The list of azimuthal orders used for searching for GYRE detail files; by default [0,0,0].
    search_n : list[int], optional
        The list of radial orders used for searching for GYRE detail files; by default [23,24,25].
    """

    # initialization method
    def __init__(
        self, search_l=[1, 1, 1], search_m=[0, 0, 0], search_n=[23, 24, 25]
    ) -> None:
        # initialize the superclass
        super().__init__(
            search_elements={
                'l': search_l,
                'm': search_m,
                'n': search_n,
                'mode': lambda x: 'g_mode' if x else 'p_mode',  # type: ignore
            },
            pattern_dictionary={
                'l': re.compile(r'.*l(\d+).*'),
                'm': re.compile(r'.*m([\+|-])(\d+).*'),
                'n': re.compile(r'.*n\+(\d+).*'),
                'mode': lambda x: re.compile(r'.*(g_mode).*')
                if x
                else re.compile(r'.*(p_mode).*'),  # type: ignore
            },
        )

    # callable method used to assess conditions for sub-selection
    def subselect(
        self,
        name: str,
        additional_substring: list[str] | tuple[str, ...] | None = None,
        mode_number: int = 1,
        g_mode: bool = True,
    ) -> bool:
        """Method used to sub-select between different GYRE detail files.

        Parameters
        ----------
        name : str
            The name of the file to be used for sub-selection.
        additional_substring : list[str] or str or None, optional
            An additional substring used for additional sub-selection. (Needs to be present in the file name.) Default: None.
        mode_number : int, optional
            The number of the mode for which you are trying to obtain/read files; by default 1.
        g_mode : bool, optional
            If True, search for g modes. If False, search for other modes; by default True.

        Returns
        -------
        bool
            The sub-selection bool.
        """
        return self._verify_conditions(
            additional_substring,
            name=name,
            element_nr=mode_number - 1,
            g_mode=g_mode,
        )

    # method used to verify you verify one of the conditions for the three mode files
    @singledispatchmethod
    def _verify_conditions(
        self,
        additional_substring: list[str] | tuple[str, ...] | None,
        name: str,
        element_nr: int,
        g_mode: bool,
    ) -> bool:
        """Internal dispatched utility method used to match one of the three conditions specified for the mode files.

        Parameters
        ----------
        additional_substring : list[str] or str or None
            An additional substring used for additional sub-selection. (Needs to be present in the file name.)
        name : str
            The name of the file to be used for sub-selection.
        element_nr : int
            The number of the mode for which you are trying to obtain/read files.
        g_mode : bool
            If True, search for g modes. If False, search for other modes.

        Returns
        -------
        bool
            The sub-selection bool.
        """
        raise NotImplementedError(
            "Could not determine the type of 'additional_substring'."
        )

    @_verify_conditions.register
    def _(self, additional_substring: list, name, element_nr, g_mode):
        # check the additional substring, if necessary
        if additional_substring[element_nr] not in name:
            return False
        # loop over the elements of the search dictionary
        for _key_search, _condition_element in self._search_elements.items():
            # break early if one of the statements is not fulfilled
            if not self._conditional(
                _condition_element, _key_search, name, element_nr, g_mode
            ):
                return False
        else:
            # no breaks: select this file whose name meets the conditions
            return True

    @_verify_conditions.register
    def _(self, additional_substring: None, name, element_nr, g_mode):
        # loop over the elements of the search dictionary
        for _key_search, _condition_element in self._search_elements.items():
            # break early if one of the statements is not fulfilled
            if not self._conditional(
                _condition_element, _key_search, name, element_nr, g_mode
            ):
                return False
        else:
            # no breaks: select this file whose name meets the conditions
            return True

    @_verify_conditions.register
    def _(self, additional_substring: str, name, element_nr, g_mode):
        # check the additional substring, if necessary
        if additional_substring not in name:
            return False
        # loop over the elements of the search dictionary
        for _key_search, _condition_element in self._search_elements.items():
            # break early if one of the statements is not fulfilled
            if not self._conditional(
                _condition_element, _key_search, name, element_nr, g_mode
            ):
                return False
        else:
            # no breaks: select this file whose name meets the conditions
            return True

    @_verify_conditions.register
    def _(self, additional_substring: tuple, name, element_nr, g_mode):
        # check the additional substrings stored in a tuple, if necessary
        for _a_s in additional_substring:
            if _a_s not in name:
                return False
        # loop over the elements of the search dictionary
        for _key_search, _condition_element in self._search_elements.items():
            # break early if one of the statements is not fulfilled
            if not self._conditional(
                _condition_element, _key_search, name, element_nr, g_mode
            ):
                return False
        else:
            # no breaks: select this file whose name meets the conditions
            return True

    # internal utility method that verifies conditions for sub-selection
    def _conditional(
        self,
        conditional_element: list | tuple,
        element_key: str,
        name: str,
        element_nr: int,
        g_mode: bool,
    ):
        """Dispatched internal method used for conditional sub-selection of files.

        Parameters
        ----------
        conditional_element : list or tuple
            The condition used to sub-select files.
        element_key: str
            The keyword of the conditional element.
        name : str
            The name of the file that is subjected to conditional tests used to sub-select files.
        element_nr : int
            The number of the mode for which you are trying to obtain/read files.
        g_mode : bool
            If True, search for g modes. If False, search for other modes.
        """
        # retrieve the pattern
        _my_pattern = self._pattern_dictionary[element_key]
        # distinguish between actions taken for 'm' and for 'l' and 'n' conditional elements
        if element_key == 'm':
            # extract the information on the 'm' argument from the name,  handle sign of 'm' during the check, and handle non-matches
            if _match := _my_pattern.match(name):
                if '+' == _match.group(1):
                    # positive sign
                    return (
                        int(_match.group(2)) == conditional_element[element_nr]
                    )
                else:
                    # negative sign
                    return (-1) * int(_match.group(2)) == conditional_element[
                        element_nr
                    ]
            else:
                return False
        elif element_key == 'mode':
            # extract the callable on the 'mode' argument from the name
            modecal = conditional_element(g_mode)  # type: ignore
            # extract the actual pattern checking function
            _my_pattern = _my_pattern(g_mode)  # type: ignore
            # perform matching
            if _match := _my_pattern.match(name):
                return _match.group(1) == modecal
            else:
                return False
        else:
            # extract the information on the 'l' or 'n' argument from the name, and check if it is in the list + return the result of that check
            if my_match := _my_pattern.match(name):
                return int(my_match.group(1)) == conditional_element[element_nr]
            else:
                return False


# class used to sub-select between polytrope model files
class PolytropeModelSubSelector(SubSelectorFiles):
    """Python class that holds methods used to sub-select between polytrope model files.

    Parameters
    ----------
    polytrope_model_suffix : str, optional
        The suffix of the polytrope model files.
    """

    # initialization method
    def __init__(self, polytrope_model_suffix='h5') -> None:
        # initialize the superclass
        super().__init__(
            search_elements={'file_suffix': (polytrope_model_suffix, True)},
            pattern_dictionary={
                'file_suffix': re.compile(rf'.+\.({polytrope_model_suffix})')
            },
        )


# class used to sub-select between polytrope oscillation files
class PolytropeOscillationModelSubSelector(SubSelectorFiles):
    """Python class that holds methods used to sub-select between polytrope oscillation model files.

    Parameters
    ----------
    search_l : list[int], optional
        The list of spherical degrees used for searching for polytrope oscillation files; by default [1,1,1].
    search_m : list[int], optional
        The list of azimuthal orders used for searching for polytrope oscillation files; by default [0,0,0].
    search_n : list[int], optional
        The list of radial orders used for searching for polytrope oscillation files; by default [23,24,25].
    """

    # initialization method
    def __init__(
        self, search_l=[1, 1, 1], search_m=[0, 0, 0], search_n=[23, 24, 25]
    ) -> None:
        # initialize the superclass
        super().__init__(
            search_elements={
                'l': search_l,
                'm': search_m,
                'n': search_n,
                'mode': lambda x: 'g_mode' if x else 'p_mode',  # type: ignore
            },
            pattern_dictionary={
                'l': re.compile(r'.*l(\d+).*'),
                'm': re.compile(r'.*m([\+|-])(\d+).*'),
                'n': re.compile(r'.*n\+(\d+).*'),
                'mode': lambda x: re.compile(r'.*(g_mode).*')
                if x
                else re.compile(r'.*(p_mode).*'),  # type: ignore
            },
        )

    # callable method used to assess conditions for sub-selection
    def subselect(
        self, name, additional_substring=None, mode_number=1, g_mode=True
    ):
        """Method used to sub-select between different GYRE detail files.

        Parameters
        ----------
        name : str
            The name of the file to be used for sub-selection.
        additional_substring : list[str] or str or None, optional
            An additional substring used for additional sub-selection. This substring needs to be present in the file name; by default None.
        mode_number : int, optional
            The number of the mode for which you are trying to obtain/read files; by default 1.
        g_mode : bool, optional
            If True, search for g modes. If False, search for other modes; by default True.

        Returns
        -------
        bool
            The sub-selection bool.
        """
        return self._verify_conditions(
            additional_substring,
            name=name,
            element_nr=mode_number - 1,
            g_mode=g_mode,
        )

    # method used to verify you verify one of the conditions for the three mode files
    @singledispatchmethod
    def _verify_conditions(
        self, additional_substring, name, element_nr, g_mode
    ):
        """Internal dispatched utility method used to match one of the three conditions specified for the mode files.

        Parameters
        ----------
        additional_substring : list[str] or str or None
            An additional substring used for additional sub-selection. (Needs to be present in the file name.)
        name : str
            The name of the file to be used for sub-selection.
        element_nr : int
            The number of the mode for which you are trying to obtain/read files.
        g_mode : bool
            If True, search for g modes. If False, search for other modes.

        Returns
        -------
        bool
            The sub-selection bool.
        """
        raise NotImplementedError(
            'Could not determine the type of ' "'additional_substring'."
        )

    @_verify_conditions.register
    def _(self, additional_substring: list, name, element_nr, g_mode):
        # check the additional substring, if necessary
        if additional_substring[element_nr] not in name:
            return False
        # loop over the elements of the search dictionary
        for _key_search, _condition_element in self._search_elements.items():
            # break early if one of the statements is not fulfilled
            if not self._conditional(
                _condition_element, _key_search, name, element_nr, g_mode
            ):
                return False
        else:
            # no breaks: select this file whose name meets the conditions
            return True

    @_verify_conditions.register
    def _(self, additional_substring: None, name, element_nr, g_mode):
        # loop over the elements of the search dictionary
        for _key_search, _condition_element in self._search_elements.items():
            # break early if one of the statements is not fulfilled
            if not self._conditional(
                _condition_element, _key_search, name, element_nr, g_mode
            ):
                return False
        else:
            # no breaks: select this file whose name meets the conditions
            return True

    @_verify_conditions.register
    def _(self, additional_substring: str, name, element_nr, g_mode):
        # check the additional substring, if necessary
        if additional_substring not in name:
            return False
        # loop over the elements of the search dictionary
        for _key_search, _condition_element in self._search_elements.items():
            # break early if one of the statements is not fulfilled
            if not self._conditional(
                _condition_element, _key_search, name, element_nr, g_mode
            ):
                return False
        else:
            # no breaks: select this file whose name meets the conditions
            return True

    @_verify_conditions.register
    def _(self, additional_substring: tuple, name, element_nr, g_mode):
        # check the additional substrings stored in a tuple, if necessary
        for _a_s in additional_substring:
            if _a_s not in name:
                return False
        # loop over the elements of the search dictionary
        for _key_search, _condition_element in self._search_elements.items():
            # break early if one of the statements is not fulfilled
            if not self._conditional(
                _condition_element, _key_search, name, element_nr, g_mode
            ):
                return False
        else:
            # no breaks: select this file whose name meets the conditions
            return True

    # internal utility method that verifies conditions for sub-selection
    def _conditional(
        self,
        conditional_element: list | tuple,
        element_key: str,
        name: str,
        element_nr: int,
        g_mode: bool,
    ) -> bool:
        """Dispatched internal method used for conditional sub-selection of files.

        Parameters
        ----------
        conditional_element : list or tuple
            The condition used to sub-select files.
        element_key: str
            The keyword of the conditional element.
        name : str
            The name of the file that is subjected to conditional tests used to sub-select files.
        element_nr : int
            The number of the mode for which you are trying to obtain/read files.
        g_mode : bool
            If True, search for g modes. If False, search for other modes.

        Returns
        -------
        bool
            The condition check boolean.
        """
        # retrieve the pattern
        _my_pattern = self._pattern_dictionary[element_key]
        # distinguish between actions taken for 'm' and for 'l' and 'n' conditional elements
        if element_key == 'm':
            # extract the information on the 'm' argument from the name, handle sign of 'm' during the check, and handle non-matches
            if _match := _my_pattern.match(name):
                if '+' == _match.group(1):
                    # positive sign
                    return (
                        int(_match.group(2)) == conditional_element[element_nr]
                    )
                else:
                    # negative sign
                    return (-1) * int(_match.group(2)) == conditional_element[
                        element_nr
                    ]
            else:
                return False
        elif element_key == 'mode':
            if TYPE_CHECKING:
                assert isinstance(conditional_element, Callable)
            # extract the callable on the 'mode' argument from the name
            modecal = conditional_element(g_mode)
            # extract the actual pattern checking function
            _my_pattern = _my_pattern(g_mode)  # type: ignore
            # perform matching
            if _match := _my_pattern.match(name):
                return _match.group(1) == modecal
            else:
                return False
        else:
            # extract the information on the 'l' or 'n' argument from the name, and check if it is in the list + return the result of that check
            if my_match := _my_pattern.match(name):
                return int(my_match.group(1)) == conditional_element[element_nr]
            else:
                return False
