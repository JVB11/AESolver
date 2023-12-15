"""Python module containing class that generates a name for a mode data file that needs to be loaded to for example generate plots, as well as a function that generates a list of such load file names.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import logging
import sys
from multimethod import multimethod

# type checking imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any
    from collections.abc import Sequence


# set up logger
logger = logging.getLogger(__name__)


# function that generates a list of paths of files to be loaded
def generate_load_file_path_list(load_dict: 'dict[str, str | int | Sequence[str | list[str]]]') -> 'Sequence[str | list[str]]':
    """Function generating a list of paths of files that need to be loaded by using the custom class 'LoadFileName'.

    Parameters
    ----------
    my_load_dict : dict[str, str | int | Sequence[str | list[str]]]
        Contains all necessary information to construct a list of load file paths. (Values behind keys are lists.)

    Returns
    -------
    Sequence[str | list[str]]
        The list of paths to files whose data need to be loaded.
    """
    # return the list of file paths
    try:
        # get the number of files in the loaded dictionary
        nr_files = load_dict['nr_files']
        if TYPE_CHECKING:
            assert isinstance(nr_files, int)
        # return the list of file paths
        return [
            LoadFileName(load_dict, _i)()
            for _i in range(nr_files)
        ]
    except KeyError:
        logger.exception(
            'There is some missing information in the loading dictionary. Please check your inlist input values. Now exiting.'
        )
        sys.exit()


# class that generates the name of a 'loadfile'
class LoadFileName:
    """Generates the name of a file that needs to be loaded.

    Parameters
    ----------
    my_load_dict : dict
        Contains all necessary information to construct a file name. Data linked to the keys are lists, so that an 'index_number' is necessary to generate the load name of a specific file.
    index_number : int
        Necessary index for information in 'my_load_dict'.
    dec_sep : str, optional
        Denotes the decimal separator used (in string format); by default '.'.

    Returns
    -------
    str
        The formatted filename.
    """

    # attribute type declarations
    _raw_info: dict
    _index: int
    _dec_sep: str
    _base_dir: str
    _output_dir: str
    _use_standard_format: bool
    _base_name: str | list[str]
    _spec_name: str | list[str]
    _mer_ord: str
    _az_ord: str
    _suffix: str
    _mass: str
    _xc: str
    _rad_ord: str
    _generic_name: str | list[str]

    def __init__(
        self, my_loading_dict: dict, index_number: int, dec_sep: str = '.'
    ) -> None:
        # store the raw information used to access data in the dictionary
        self._raw_info = my_loading_dict
        self._index = index_number
        self._dec_sep = dec_sep
        # store base dir and output dir
        if isinstance((base_dir := my_loading_dict['base_dir']), str) and isinstance((output_dir := my_loading_dict['output_dir']), str):
            self._base_dir = base_dir
            self._output_dir = output_dir
        else:
            raise TypeError("Base_dir and output_dir should be of the String type in order to load file data.")
        # unpack the information in the loading dictionary
        self._check_if_standard_format()
        self._unpack_dictionary_string_info()

    def __call__(self) -> str | list[str]:
        # differentiate between the two paths
        if isinstance(my_output := self.generate_generic_name(self._base_name, self._spec_name, self._suffix), list):
            # return list of strings
            return [f'{self._base_dir}/{self._output_dir}/{_generic_name}' for _generic_name in my_output]
        else:
            return f'{self._base_dir}/{self._output_dir}/{my_output}'

    @multimethod
    def generate_generic_name(self, base_file_name_or_names: str, specification_name_or_names: str, suffix_or_suffixes: str) -> str:
        # generate the generic string variable that represent the name of the file to be loaded
        generic_name = f'{base_file_name_or_names}{f"_{specification_name_or_names}" if len(specification_name_or_names) > 0 else ""}'
        # check if standard format info should be added to string
        if self._use_standard_format:
            # add standard format information --> mass, Xc, radial order info
            generic_name += f'_M{self._mass}_Xc{self._xc}_{self._rad_ord}'
        # add meridional degree and azimuthal order info
        generic_name += f'_{self._mer_ord}_{self._az_ord}'
        # return the generic name string
        return f'{generic_name}.{suffix_or_suffixes}'

    @multimethod
    def generate_generic_name(self, base_file_name_or_names: list[str], specification_name_or_names: list[str], suffix_or_suffixes: list[str]) -> list[str]:
        # initialize the list of generic names
        generic_names = []
        # generate the list of generic names
        for _bn, _sn, _sf in zip(base_file_name_or_names, specification_name_or_names, suffix_or_suffixes):
            # generate the generic string variable that represent the name of the file to be loaded
            generic_name = f'{_bn}{f"_{_sn}" if len(_sn) > 0 else ""}'
            # check if standard format info should be added to string
            if self._use_standard_format:
                # add standard format information --> mass, Xc, radial order info
                generic_name += f'_M{self._mass}_Xc{self._xc}_{self._rad_ord}'
            # add meridional degree and azimuthal order info
            generic_name += f'_{self._mer_ord}_{self._az_ord}'
            # append to list
            generic_names.append(f'{generic_name}.{_sf}')
        # return the generic name string
        return generic_names

    def _check_if_standard_format(self) -> None:
        """Stores whether a (long) standard format name is used or not."""
        self._use_standard_format = self._raw_info['use_standard_format']

    def _unpack_dictionary_string_info(self) -> None:
        """Unpacks the dictionary information used to generate the loading file paths into attributes of this class. Only unpacks the necessary information for generating the required type of name: custom or standard format."""
        # retrieve all necessary information for short format names
        self._base_name = self._access_raw_info('base_file_names')
        self._spec_name = self._access_raw_info('specification_names')
        self._mer_ord = self._meridional_order_string()
        self._az_ord = self._azimuthal_order_string()
        self._suffix = self._access_raw_info('file_suffix')
        # retrieve additional information for (long) standard format
        if self._use_standard_format:
            self._mass = self._access_raw_info('masses')
            self._xc = self._split_xc_value()
            self._rad_ord = self._radial_order_string()

    def _access_raw_info(self, my_property: str) -> "Any":
        """Generic method used to access raw information from the dictionary.

        Parameters
        ----------
        my_property : str
            Denotes the property for which you want to access raw info.
        """
        return self._raw_info[my_property][self._index]

    def _split_xc_value(self) -> str:
        """Retrieves and formats the Xc value for the load file.

        Returns
        -------
        str
            The formatted string for the Xc value.
        """
        i, f = str(self._access_raw_info('Xcs')).split(self._dec_sep)
        return f'{i}_{f}'

    def _radial_order_string(self) -> str:
        """Generates a radial order string based on a triple pair input.

        Returns
        -------
        str
            The formatted string with information on the radial order.
        """
        # obtain the triple pair
        triple_pair = self._access_raw_info('triple_pairs')
        # return the formatted string
        return f'd{triple_pair[0][0]}_{triple_pair[0][1]}_d{triple_pair[1][0]}_{triple_pair[1][1]}_p{triple_pair[2][0]}_{triple_pair[2][1]}'

    def _order_string(self, pre_term: str, access_term: str) -> str:
        """Generic method used to generate a formatted string for a specific order/degree.

        Notes
        -----
        Assumes you are dealing with quadratic (triad) coupling!

        Parameters
        ----------
        pre_term : str
            The (formatted) string to be put in front of the formatted raw data.
        access_term : str
            The string used to access the raw data.

        Returns
        -------
        str
            A formatted string that contains the accessed information.
        """
        # obtain the raw information
        _raw_info = self._access_raw_info(access_term)
        # return the formatted string
        return f'{pre_term}_{_raw_info[0]}_{_raw_info[1]}_{_raw_info[2]}'

    def _meridional_order_string(self) -> str:
        """Method used to generate a formatted string with information on the meridional degrees.

        Returns
        -------
        str
            Formatted string containing information on the meridional degrees.
        """
        # return the formatted string
        return self._order_string(
            pre_term='k', access_term='meridional_degrees'
        )

    def _azimuthal_order_string(self) -> str:
        """Method used to generate a formatted string with information on the azimuthal orders.

        Returns
        -------
        str
            Formatted string containing information on the azimuthal orders.
        """
        # return the formatted string
        return self._order_string(pre_term='m', access_term='azimuthal_orders')
