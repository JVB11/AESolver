"""Initialization file for the python module that contains functions that can be used to determine optimal bin width for histogram representation of data.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import the functions in the module
from .blas_jacky import (
    hogg_jackknife_likelihood_blas as histogram_likelihood_blas,
)
from .blas_jacky import (
    hogg_jackknife_likelihood_blas_log as log_histogram_likelihood_blas,
)
from .blas_jacky import histogram_blas_spec_width, histogram_blas_log_spec_edges
from .numexpr_jacky import (
    hogg_jackknife_likelihood_numexpr as histogram_likelihood_numexpr,
)
from .numexpr_jacky import (
    hogg_jackknife_likelihood_numexpr_log as log_histogram_likelihood_numexpr,
)
from .numexpr_jacky import (
    histogram_numexpr_log_spec_edges,
    histogram_numexpr_spec_width,
)
from .numpy_iterator_jacky import (
    hogg_jackknife_likelihood_iterator as histogram_likelihood_iter,
)
from .numpy_iterator_jacky import (
    hogg_jackknife_likelihood_iterator_log as log_histogram_likelihood_iter,
)
from .numpy_iterator_jacky import (
    hogg_jackknife_likelihood_numpy as histogram_likelihood_numpy,
)
from .numpy_iterator_jacky import histogram_spec_width, histogram_log_spec_edges
from .shared import histogram_log_counts


# guard imports from partially initialized module while using pytest (this means that the cython modules cannot directly be tested)
try:
    from .cython_jacky import histogram_likelihood_cython
    from .cython_jacky import log_histogram_likelihood_cython
    from .cython_jacky import (
        histogram_cython_spec_width,
        histogram_cython_log_spec_edges,
    )

    __all__ = [
        'histogram_likelihood_numpy',
        'histogram_likelihood_iter',
        'log_histogram_likelihood_iter',
        'histogram_likelihood_blas',
        'log_histogram_likelihood_blas',
        'histogram_blas_spec_width',
        'histogram_blas_log_spec_edges',
        'histogram_likelihood_numexpr',
        'log_histogram_likelihood_numexpr',
        'histogram_numexpr_log_spec_edges',
        'histogram_numexpr_spec_width',
        'histogram_log_counts',
        'histogram_spec_width',
        'histogram_log_spec_edges',
        'histogram_likelihood_cython',
        'log_histogram_likelihood_cython',
        'histogram_cython_spec_width',
        'histogram_cython_log_spec_edges',
    ]
except ImportError:
    __all__ = [
        'histogram_likelihood_numpy',
        'histogram_likelihood_iter',
        'log_histogram_likelihood_iter',
        'histogram_likelihood_blas',
        'log_histogram_likelihood_blas',
        'histogram_blas_spec_width',
        'histogram_blas_log_spec_edges',
        'histogram_likelihood_numexpr',
        'log_histogram_likelihood_numexpr',
        'histogram_numexpr_log_spec_edges',
        'histogram_numexpr_spec_width',
        'histogram_log_counts',
        'histogram_spec_width',
        'histogram_log_spec_edges',
    ]


# version + author + functions
__version__ = '1.0.0'
__author__ = 'Jordan Van Beeck'
