import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport log
from cython.parallel cimport prange


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef _cython_jacky_serial_loop(Py_ssize_t bin_view_size, cnp.float64_t[::1] bin_data_view, double alpha, double bw, double empirical_probability_conversion_fac):
    # ensure cython compiler knows that the counter in the prange loop is a C integer
    cdef int d
    # create the output jackknife likelihood variable
    cdef cnp.float64_t jacky = 0.0
    # loop to fill binning array using view
    for d in range(bin_view_size):
        jacky += bin_data_view[d] * log((bin_data_view[d] + alpha - 1.0) / (empirical_probability_conversion_fac * bw))
    # return the jackknife likelihood
    return jacky


def cython_jacky_serial_loop(cnp.float64_t[::1] bin_data_view, double alpha, double bw, double empirical_probability_conversion_fac):
    # store input size
    cdef Py_ssize_t view_size = bin_data_view.shape[0]
    # obtain and return the jackknife likelihood value
    return _cython_jacky_serial_loop(view_size, bin_data_view, alpha, bw, empirical_probability_conversion_fac)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef _cython_variable_bw_jacky_serial_loop(Py_ssize_t view_size, cnp.float64_t[::1] bin_data_view, cnp.float64_t[::1] my_bin_edges, double alpha, double empirical_probability_conversion_fac):
    # ensure cython compiler knows that the counter in the prange loop is a C integer
    cdef int d
    # create the output jackknife likelihood variable
    cdef cnp.float64_t jacky = 0.0
    # loop to fill binning array using view
    for d in prange(view_size, nogil=True):
        jacky += bin_data_view[d] * log((bin_data_view[d] + alpha - 1.0) / (empirical_probability_conversion_fac * (my_bin_edges[d + 1] - my_bin_edges[d])))
    # return the jackknife likelihood
    return jacky


def cython_variable_bw_jacky_serial_loop(cnp.float64_t[::1] bin_data_view, cnp.float64_t[::1] my_bin_edges, double alpha, double empirical_probability_conversion_fac):
    # store input size
    cdef Py_ssize_t view_size = bin_data_view.shape[0]
    # obtain and return the jackknife likelihood value
    return _cython_variable_bw_jacky_serial_loop(view_size, bin_data_view, my_bin_edges, alpha, empirical_probability_conversion_fac)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef _cython_jacky_parallel_loop(Py_ssize_t view_size, cnp.float64_t[::1] bin_data_view, double alpha, double bw, double empirical_probability_conversion_fac):
    # ensure cython compiler knows that the counter in the prange loop is a C integer
    cdef int d
    # create the output jackknife likelihood variable
    cdef cnp.float64_t jacky = 0.0
    # loop to fill binning array using view
    for d in range(view_size):
        jacky += bin_data_view[d] * log((bin_data_view[d] + alpha - 1.0) / (empirical_probability_conversion_fac * bw))
    # return the jackknife likelihood
    return jacky


def cython_jacky_parallel_loop(cnp.float64_t[::1] bin_data_view, double alpha, double bw, double empirical_probability_conversion_fac):
    # store input size
    cdef Py_ssize_t view_size = bin_data_view.shape[0]
    # obtain and return the jackknife likelihood value
    return _cython_jacky_parallel_loop(view_size, bin_data_view, alpha, bw, empirical_probability_conversion_fac)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef _cython_variable_bw_jacky_parallel_loop(Py_ssize_t view_size, cnp.float64_t[::1] bin_data_view, cnp.float64_t[::1] my_bin_edges, double alpha, double empirical_probability_conversion_fac):
    # ensure cython compiler knows that the counter in the prange loop is a C integer
    cdef int d
    # create the output jackknife likelihood variable
    cdef cnp.float64_t jacky = 0.0
    # loop to fill binning array using view
    for d in prange(view_size, nogil=True):
        jacky += bin_data_view[d] * log((bin_data_view[d] + alpha - 1.0) / (empirical_probability_conversion_fac * (my_bin_edges[d + 1] - my_bin_edges[d])))
    # return the jackknife likelihood
    return jacky


def cython_variable_bw_jacky_parallel_loop(cnp.float64_t[::1] bin_data_view, cnp.float64_t[::1] my_bin_edges, double alpha, double empirical_probability_conversion_fac):
    # store input size
    cdef Py_ssize_t view_size = bin_data_view.shape[0]
    # obtain and return the jackknife likelihood value
    return _cython_variable_bw_jacky_parallel_loop(view_size, bin_data_view, my_bin_edges, alpha, empirical_probability_conversion_fac)


def cython_jacky_loop(Py_ssize_t view_size, cnp.float64_t[::1] bin_data_view, double alpha, double bw, double empirical_probability_conversion_fac, bint use_parallel):
    # obtain and return the jackknife likelihood   
    if use_parallel:
        return _cython_jacky_parallel_loop(view_size, bin_data_view, alpha, bw, empirical_probability_conversion_fac)
    else:
        return _cython_jacky_serial_loop(view_size, bin_data_view, alpha, bw, empirical_probability_conversion_fac)


def cython_variable_bw_jacky_loop(Py_ssize_t view_size, cnp.float64_t[::1] bin_data_view, cnp.float64_t[::1] my_bin_edges, double alpha, double empirical_probability_conversion_fac, bint use_parallel):
    # obtain and return the jackknife likelihood   
    if use_parallel:
        return _cython_variable_bw_jacky_parallel_loop(view_size, bin_data_view, my_bin_edges, alpha, empirical_probability_conversion_fac)
    else:
        return _cython_variable_bw_jacky_serial_loop(view_size, bin_data_view, my_bin_edges, alpha, empirical_probability_conversion_fac)


def cython_generalized_jacky_loop(cnp.float64_t[::1] bin_data_view, double alpha, double empirical_probability_conversion_fac, bint use_parallel, cnp.float64_t[::1] my_bin_edges, double bw, bint constant_bw):
    # store input size
    cdef Py_ssize_t view_size = bin_data_view.shape[0]
    # make a selection of the different methods based on input parameters
    if constant_bw:
        return cython_jacky_loop(view_size, bin_data_view, alpha, bw, empirical_probability_conversion_fac, use_parallel)
    else:
        return cython_variable_bw_jacky_loop(view_size, bin_data_view, my_bin_edges, alpha, empirical_probability_conversion_fac, use_parallel)
