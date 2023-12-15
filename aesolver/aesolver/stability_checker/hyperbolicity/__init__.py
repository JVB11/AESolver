"""Python initialization file for the package that verifies the hyperbolicity of the stationary solutions of the amplitude equations.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# custom imports
from .hyperbolicity_checker import check_hyperbolicity_of_fp as check_hyper
from .hyperbolicity_checker import (
    jacobian_in_fp_real_four_ae_system as hyper_jac,
)
from .hyperbolicity_checker import (
    jacobian_in_fp_real_three_ae_system as hyper_jac_three,
)


# define version, author and auto-import packages
__version__ = '1.0.0'
__author__ = 'Jordan Van Beeck'
__all__ = ['check_hyper', 'hyper_jac', 'hyper_jac_three']
