"""Utility module containing functions used to resolve paths to files.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
import errno
import os
from pathlib import Path


def _get_abspath_to_run(sys_arguments: list) -> str:
    """Get the absolute path to the run file directory.

    Parameters
    ----------
    sys_arguments : list
        System arguments list.

    Returns
    -------
    str
        Absolute path to the run file directory.
    """
    return os.path.abspath(os.path.dirname(sys_arguments[0]))


def resolve_path_to_file(
    sys_arguments: list,
    file_name: str,
    default_path: str,
    default_run_path: str = '..',
) -> Path:
    """Resolves the path to a file with a known default relative path from the base directory.

    Notes
    -----
    Used to hardcode relative paths in Python packages/modules.

    Parameters
    ----------
    sys_arguments : sys.argv
        The input arguments used when running the script.
    file_name : str
        Name of the file for which you would like to obtain the resolved path.
    default_path : str
        The default relative path to the base directory, which is used to ultimately resolve the path from the current work directory.
    default_run_path : str, optional
        Conversion from the run directory to the base directory; by default '..'.

    Returns
    -------
    Path
        Resolved path to the file that you wanted to obtain.

    Raises
    ------
    FileNotFoundError
        Raised when the corresponding file cannot be found in the programmatic paths. This is likely due to mis-specification of the 'default_path'!
    """
    if (my_default_path := Path(f'{default_path}/{file_name}')).exists():
        # return the resolved (default) path
        return my_default_path.resolve()
    else:
        # get the path to the file directory containing the run file
        my_run_file_path = Path(_get_abspath_to_run(sys_arguments=sys_arguments))
        if (
            my_full_path := (
                my_run_file_path / default_run_path / my_default_path
            ).resolve()
        ).exists():
            return my_full_path
        else:
            raise FileNotFoundError(
                errno.ENOENT,
                os.strerror(errno.ENOENT),
                f"{file_name} (searched for this file on the following paths: '{my_default_path}' and '{my_full_path})'",
            )
