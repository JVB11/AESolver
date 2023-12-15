"""Initialization file for the num_integrator library.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""

__version__ = '1.0.0'
__author__ = 'Jordan Van Beeck'


from . import libcpp_num_integrator as cni  # type: ignore


__all__ = ['cni']
