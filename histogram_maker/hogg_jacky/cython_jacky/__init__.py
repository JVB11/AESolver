"""Python initialization file for the cython-powered sub-module.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import typing information
from .cy_jacky import (
    hogg_jackknife_likelihood_cython as histogram_likelihood_cython,
)
from .cy_jacky import (
    hogg_jackknife_likelihood_cython_log as log_histogram_likelihood_cython,
)
from .cy_jacky import (
    histogram_cython_spec_width,
    histogram_cython_log_spec_edges,
)


# make extra information available
__version__ = '1.0.0'
__author__ = 'Jordan Van Beeck'
__all__ = [
    'histogram_likelihood_cython',
    'log_histogram_likelihood_cython',
    'histogram_cython_spec_width',
    'histogram_cython_log_spec_edges',
]
