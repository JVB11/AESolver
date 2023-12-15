"""Python enumeration module for the HDF5-format saver.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
from enum import Enum


# HDF5 file open mode selection
class Hdf5FileOpenModeSelector(Enum):
    """Enumeration class containing data that select the file open mode."""

    # define the strings that define which file opening actions will be undertaken
    write = 'w'
    app = 'a'
    rw = 'r+'

    # define the class method that selects the specific file opening mode
    @classmethod
    def select_file_open_mode(cls, r_w: bool = False, app: bool = False) -> str:
        """Method that selects the file open mode based on the save type.

        Notes
        -----
        If both 'r_w' and 'app' are False, the file will open in write/'w' mode.

        Parameters
        ----------
        r_w : bool, optional
            Defines whether files should be opened in read / write mode.
        app : bool, optional
            Defines whether files should be opened in append mode.

        Returns
        -------
        str
            Denotes the file opening mode.
        """
        # check if the string is in the members of this enumeration class to determine what file open mode is chosen
        if r_w:
            # read/write mode
            return cls.rw.value
        elif app:
            # append mode
            return cls.app.value
        else:
            # write mode
            return cls.write.value
