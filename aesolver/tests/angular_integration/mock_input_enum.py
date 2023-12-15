'''Contains code to generate mock input data for our test of this package.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
'''
# import statements
import typing
# import custom pytest utility class
from pytest_util_classes import EVs


# generate a subclass used to generate input data
class InputDataEnum(EVs):
    """Subclass used to generate mock-up input data."""
    
    # get representation string
    @classmethod
    def get_custom_repr(cls) -> str:
        """Get custom representation name for the Input class,
        which consists of the name of the class where the
        'InputData' part has been removed.
        """
        # remove 'InputData' from the classname to return the
        # custom representation of its name
        return f"{cls.__name__[9:]}"
    
    @classmethod
    def _get_translation_dict(
        cls, max_nr_data: int=3
        ) -> dict[str, typing.Any]:
        """Get dictionary translating input data keys to keys to be used for storage.
        
        Parameters
        ----------
        max_nr_data : int, optional
            The maximum number of numpy array input data to be used for the angular integration tests; by default 3.
        """
        # obtain the translation dict for
        # the input data arrays
        DATA: dict = {
            f'MY_DATA_ARR{_i}': f'my_data_arr{_i}'
            for _i in range(1, 1 + max_nr_data)
            }
        # obtain the non-input data array translations
        NON_DATA: dict = {
            'MU_VALUES': '_mu_values',
            'THETA_VALUES': '_theta_values'
        }
        # return the translation dict
        return DATA | NON_DATA
    
    @classmethod
    def access_all_input_data(
        cls, max_nr_data: int=3
        ) -> dict[str, typing.Any]:
        """Method used to access all input data stored in the enum, using a translation dictionary defined for each separate class.
        
        Notes
        -----
        Max. value of max_nr_data = 3.
        
        Parameters
        ----------
        max_nr_data : int, optional
            The maximum number of numpy array input data to be used for the angular integration tests; by default 3.
            
        Returns
        -------
        dict
            Input data stored in a dictionary.
        """
        # get the enumeration dictionary
        my_enum_dict: dict = cls.get_enumeration_dict()
        # get the translation dictionary
        my_translation_dict: dict = cls._get_translation_dict(
            max_nr_data=max_nr_data
        )
        # return the translated input dictionary
        return {
            my_translation_dict[_k.upper()]: _v 
            for _k, _v in my_enum_dict.items() 
        }               
    