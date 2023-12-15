"""Python module containing sub-class to be used for testing.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import pytest util class
from pytest_util_classes import EVs  # type: ignore


# create the subclass
class EVsList(EVs):
    """Subclass of EVs that implements a method that generates a list of the enum values."""

    @classmethod
    def get_list(
        cls, shape_list=None, dtype_buffered=None, tuple_conversion=True
    ):
        """Returns list of enum member values.

        Returns
        -------
        list
            Contains the enum member values.
        """
        # get value list
        value_list = [
            cls.get_value(i.name, dtype_buffered=dtype_buffered)
            for (_, i) in cls.__members__.items()
        ]
        # check if conversion to tuple members is needed
        if tuple_conversion:
            for _i in range(len(value_list)):
                if isinstance(member_val := value_list[_i], list):
                    if len(member_val) > 0 and isinstance(member_val[0], list):
                        value_list[_i] = [tuple(x) for x in member_val]
        # check if shapes need to be adjusted
        if shape_list is not None and isinstance(shape_list, list):
            assert len(shape_list) == len(value_list)
            for _i, _s in enumerate(shape_list):
                if _s is not None:
                    value_list[_i] = value_list[_i].reshape(_s)
        # return the value list
        return value_list
