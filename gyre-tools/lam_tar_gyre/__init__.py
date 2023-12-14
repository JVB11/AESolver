"""Initialization file for the python module that contains a class used to compute the GYRE Laplace Tidal Equations' eigenvalues.

Author: Jordan Van Beeck <jordanvanbeek@hotmail.com>
"""
# import the lam_tar_gyre class
from .lam_tar_gyre import GyreLambdas


# define the function/method get_k as a separate entity, ready for use
gravito_inertial_k = GyreLambdas.get_k


# version + author
__version__ = '1.0.0'
__author__ = 'Jordan Van Beeck'
__all__ = ['gravito_inertial_k']
