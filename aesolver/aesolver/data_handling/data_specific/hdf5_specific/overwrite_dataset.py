"""Module containing function used to overwrite a HDF5 dataset using h5py.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import numpy as np
import h5py as h5
import logging


logger = logging.getLogger(__name__)


def overwrite_dataset(
    h5file_stream: h5.File,
    dataset_path: str,
    overwrite_value: np.ndarray | h5.Empty = h5.Empty('f'),
    dtype: np.dtype | None = None,
) -> None:
    """Internal utility method used to overwrite datasets in a HDF5 file.

    Parameters
    ----------
    h5file_stream: h5.File
        The stream to the HDF5 file in which the data must be stored.
    dataset_path: str
        Denotes the path to the dataset within the HDF5 file.
    overwrite_value: np.ndarray or h5.Empty
        Defines what will be used to fill up the freed space on H5py.
    dtype : None or np.dtype, optional
        Defines the datatype of the dataset that will be overwritten, if possible.
    """
    # - dataset already exists, overwrite it
    try:
        # obtain the previous data
        _previous_data = h5file_stream[dataset_path]
        # overwrite for empty data or when data types are the same
        if (is_ds := isinstance(_previous_data, h5.Dataset)) and (
            isinstance(overwrite_value, h5.Empty)
            or (_previous_data.dtype == overwrite_value.dtype)
        ):
            # overwrite it
            _previous_data[...] = overwrite_value
        elif not is_ds:
            logger.error(
                f'The passed dataset_path ({dataset_path}) does not link to a stored HDF5 dataset. Will not attempt to do anything out of caution.'
            )
        # delete dataset, otherwise data might be lost during saving
        else:
            # delete the dataset
            del h5file_stream[dataset_path]
            # remake the dataset
            if dtype is None:
                h5file_stream.create_dataset(
                    name=dataset_path, data=overwrite_value
                )
            else:
                h5file_stream.create_dataset(
                    name=dataset_path, data=overwrite_value, dtype=dtype
                )
    except ValueError:
        logger.error(
            'A value error was raised, which is probably raised due to differing sizes of previously stored data and the data that you want to add. Now deleting and overwriting.'
        )
        # delete the dataset
        del h5file_stream[dataset_path]
        # remake the dataset
        if dtype is None:
            h5file_stream.create_dataset(
                name=dataset_path, data=overwrite_value
            )
        else:
            h5file_stream.create_dataset(
                name=dataset_path, data=overwrite_value, dtype=dtype
            )
    except TypeError:
        logger.error(
            'A type error was raised, which is probably raised due to differing sizes of previously stored data and the data that you want to add. Now deleting and overwriting.'
        )
        # delete the dataset
        del h5file_stream[dataset_path]
        # remake the dataset
        if dtype is None:
            h5file_stream.create_dataset(
                name=dataset_path, data=overwrite_value
            )
        else:
            h5file_stream.create_dataset(
                name=dataset_path, data=overwrite_value, dtype=dtype
            )
