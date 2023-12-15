"""Python initialization file for the package containing the classes that will handle retrieving sufficient data used to plot radial and angular kernels.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# custom imports
from .radial_kernels import RadialKernelData
from .angular_kernels import AngularKernelData
from .input_user import poll_user_profile_for_combination_number

# define version, author and auto-import packages
__version__ = '0.1.4'
__author__ = 'Jordan Van Beeck'
__all__ = [
    'RadialKernelData',
    'AngularKernelData',
    'poll_user_profile_for_combination_number',
]
