"""Python module containing functionalities used to retrieve the initial guess for the eigenvalues of the Laplace Tidal Equations (LTE's) based on the GYRE approximation to the Hough functions.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import statements
import logging
import os
import sys
from pathlib import Path
from typing import Callable

# import non-standard package
from multimethod import multimethod  # type: ignore


# set up logger
logger = logging.getLogger(__name__)


# define the class used to retrieve the GYRE eigenvalues of the LTE's
class GyreLambdas:
    """Python class used to retrieve the GYRE eigenvalues of the LTE's.

    Parameters
    ----------
    base_gyre_dir : str, optional
        The base (install) directory of the system's GYRE installation; by default: '' (empty string).
    """

    # attribute type declarations
    # ---------------------------
    _gtf: Callable
    _base_gyre_dir: str

    # main initialization method
    # --------------------------
    def __init__(self, base_gyre_dir='') -> None:
        # create the path to the GYRE base directory
        self._set_base_gyre_dir(base_gyre_dir=base_gyre_dir)
        # set the gyre_tar_fit function
        self._set_gtf()
        # log info message
        logger.info('GYRE lambda retriever interface set up.')

    # initialization helper methods
    # -----------------------------
    def _set_gtf(self):
        """Internal initialization helper method that stores the gyre_tar_fit function."""
        # append the TAR directory to the system path
        sys.path.append(f'{self._base_gyre_dir}/src/tar/')
        # attempt to import and store the correct GYRE callable
        try:
            import gyre_tar_fit  # type: ignore
        except ModuleNotFoundError:
            logger.exception("The 'gyre_tar_fit' module was not found.")
            sys.exit()
        else:
            self._gtf = gyre_tar_fit  # type: ignore

    def _set_base_gyre_dir(self, base_gyre_dir=''):
        """Internal initialization helper method used to store the base GYRE directory.

        Parameters
        ----------
        base_gyre_dir : str, optional
            If any string is passed, this is assumed to be the GYRE directory. If a string of length zero is passed, the environment variable $GYRE_DIR is read. Default: '' (empty string).
        """
        # no base path provided --> infer
        if (base_gyre_dir is not None) and (len(base_gyre_dir) == 0):
            self._base_gyre_dir = os.environ['GYRE_DIR']
        # base path provided --> use in full path construction
        elif base_gyre_dir is not None:
            # check that the provided path to the directory exists
            if (my_path := Path(base_gyre_dir)).is_dir():
                if (my_path / 'src' / 'tar').is_dir():
                    self._base_gyre_dir = base_gyre_dir
                else:
                    logger.error(
                        f"The TAR subdirectory does not exist within 'base_gyre_dir' ({base_gyre_dir})."
                    )
                    sys.exit()
            else:
                logger.error(
                    f"The provided 'base_gyre_dir' ({base_gyre_dir}) is not a directory/does not exist."
                )
                sys.exit()
        else:
            logger.error(
                f"The provided 'base_gyre_dir' ({base_gyre_dir}) was not provided in a suitable way.'
            )
            sys.exit()

    # utility methods
    # ---------------
    @staticmethod
    def _sgnval(number):
        """Internal utility method used to retrieve the string sign of the number provided as input.

        Parameters
        ----------
        number : int
            The number for which the string sign must be determined.

        Returns
        -------
        str
            The string representing the sign of the input number.
        """
        return f'+{number}' if (number >= 0) else f'-{number}'

    @staticmethod
    def get_k(degree, azimuthal):
        """Internal utility method used to retrieve the mode identification parameter k (see e.g. Lee & Saio, 1997), defined by the azimuthal order and 'degree'.

        Notes
        -----
        This definition of the mode identification parameter 'k' is valid only for gravito-inertial modes.

        Parameters
        ----------
        degree : int
            The (spherical) degree of the mode.
        azimuthal : int
            The azimuthal order of the mode.

        Returns
        -------
        int
            The mode identification parameter k.
        """
        return degree - abs(azimuthal)

    def _retrieve_name_gyre_file(self, degree, azimuthal):
        """Internal method used to retrieve the correct name of the GYRE data tables used to read/interpolate the LTE eigenvalues.

        Parameters
        ----------
        degree : int
            The spherical degree of the mode.
        azimuthal : int
            The azimuthal order of the mode.

        Returns
        -------
        str
            The path to the file.
        """
        # retrieve the 'k' mode id factor
        _my_k = self.get_k(degree=degree, azimuthal=azimuthal)
        # construct and return the file name
        fn = (
            f'{self._base_gyre_dir}/data/tar/tar_fit.m{self._sgnval(azimuthal)}'
            f'.k{self._sgnval(_my_k)}.h5'
        )
        # check if the file exists and return the string path if it does
        if Path(fn).is_file():
            return fn
        else:
            logger.error(f'The requested GYRE TAR file ({fn}) does not exist!')
            sys.exit()

    # overloaded method used to obtain the GYRE estimates for
    # the eigenvalues of the LTE's
    # ------------------------------------------------------------------------------------
    @multimethod
    def get_lambda(self, degree: int, azimuthal: int, spin_factor: float):  # type: ignore
        """Overloaded method used to retrieve the GYRE eigenvalue of the LTE's for a single mode.

        Parameters
        ----------
        degree : int
            The spherical degree of the mode.
        azimuthal : int
            The azimuthal order of the mode.
        spin_factor : float
            The spin factor (2 Omega / omega_corot) of the mode.

        Returns
        -------
        list[float]
            The requested GYRE eigenvalue of the LTE's. NOTE: the minus sign differs due to definitions/conventions in Prat et al. (2019) and GYRE/Townsend(2020) or Lee(2012)/Lee & Saio (1997).
        """
        # get the name of the GYRE file
        _fn_gyre = self._retrieve_name_gyre_file(degree, azimuthal)
        # load the GYRE interpolation table
        tf = self._gtf.TarFit.load(_fn_gyre)
        # use the GYRE interpolation table to return the lambda estimate
        return tf.lam(spin_factor)

    @multimethod
    def get_lambda(  # noqa
        self, degree: list[int], azimuthal: list[int], spin_factor: list[float]
    ):
        """Overloaded method used to retrieve the GYRE eigenvalues of the LTE's for multiple modes at once.

        Parameters
        ----------
        degree : list[int]
            The list of spherical degrees of the modes.
        azimuthal : list[int]
            The list of azimuthal orders of the modes.
        spin_factor : list[float]
            The list of spin factors computed for each of the modes.

        Returns
        -------
        list[list, float]
            The list of requested GYRE eigenvalues of the LTE's.
        """
        # loop over recursively + return the result
        return [
            self.get_lambda(_d, _a, _sf)  # type: ignore
            for _d, _a, _sf in zip(degree, azimuthal, spin_factor)
        ]
