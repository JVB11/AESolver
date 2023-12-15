"""Module containing function to generate a dataset in a HDF5 file.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import h5py as h5


def create_group_path_dataset(
    h5file_stream: h5.File, group_specifier: str
) -> h5.Group:
    """Creates the group path for the dataset you are saving.

    Parameters
    ----------
    h5file_stream : h5.File
        The stream to the HDF5 file.
    group_specifier : str
        The specifier for the group path.

    Return
    ------
    my_group : h5.Group
        The HDF5 file group.
    """
    # require the group to be present and otherwise create it
    my_group = h5file_stream.require_group(name=group_specifier)
    # return the group
    return my_group
