"""Initialization file for the disc_integrator library.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""

__version__ = "1.0.0"
__author__ = "Jordan Van Beeck"


from . import libcpp_disc_integration as di  # type: ignore


__all__ = ["di"]
