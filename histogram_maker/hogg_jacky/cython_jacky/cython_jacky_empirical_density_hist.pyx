import numpy as np
cimport numpy as cnp
cimport cython
from cython.parallel cimport prange


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef _cython_empirical_density_hist(Py_ssize_t view_size, cnp.float64_t[::1] my_hist_view, double empirical_probability_conversion_factor, double bin_width):
    # ensure cython compiler knows that the counter in the prange loop is a C integer
    cdef int i
    # loop to perform empirical density conversion using constant bin width
    for i in range(view_size):
        my_hist_view[i] /= (empirical_probability_conversion_factor * bin_width)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef _cython_empirical_density_hist_int(Py_ssize_t view_size, cnp.float64_t[::1] my_hist_view, int empirical_probability_conversion_factor, double bin_width):
    # ensure cython compiler knows that the counter in the prange loop is a C integer
    cdef int i
    # loop to perform empirical density conversion
    for i in range(view_size):
        my_hist_view[i] /= (<double>empirical_probability_conversion_factor * bin_width)


def cython_empirical_density_hist(cnp.float64_t[::1] my_hist_view, double empirical_probability_conversion_factor, double bin_width):
    # obtain the size
    cdef Py_ssize_t s = my_hist_view.shape[0]
    # perform empirical density conversion
    _cython_empirical_density_hist(s, my_hist_view, empirical_probability_conversion_factor, bin_width)


def cython_empirical_density_hist_int(cnp.float64_t[::1] my_hist_view, int empirical_probability_conversion_factor, double bin_width):
    # obtain the size
    cdef Py_ssize_t s = my_hist_view.shape[0]
    # perform empirical density conversion
    _cython_empirical_density_hist_int(s, my_hist_view, empirical_probability_conversion_factor, bin_width)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef _cython_empirical_density_hist_parallel(Py_ssize_t view_size, cnp.float64_t[::1] my_hist_view, double empirical_probability_conversion_factor, double bin_width):
    # ensure cython compiler knows that the counter in the prange loop is a C integer
    cdef int i
    # loop to perform empirical density conversion
    for i in prange(view_size, nogil=True):
        my_hist_view[i] /= (empirical_probability_conversion_factor * bin_width)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef _cython_empirical_density_hist_int_parallel(Py_ssize_t view_size, cnp.float64_t[::1] my_hist_view, int empirical_probability_conversion_factor, double bin_width):
    # ensure cython compiler knows that the counter in the prange loop is a C integer
    cdef int i
    # loop to perform empirical density conversion
    for i in prange(view_size, nogil=True):
        my_hist_view[i] /= (<double>empirical_probability_conversion_factor * bin_width)


def cython_empirical_density_hist_parallel(cnp.float64_t[::1] my_hist_view, double empirical_probability_conversion_factor, double bin_width):
    # obtain the size
    cdef Py_ssize_t s = my_hist_view.shape[0]
    # perform empirical density conversion
    _cython_empirical_density_hist_parallel(s, my_hist_view, empirical_probability_conversion_factor, bin_width)


def cython_empirical_density_hist_int_parallel(cnp.float64_t[::1] my_hist_view, int empirical_probability_conversion_factor, double bin_width):
    # obtain the size
    cdef Py_ssize_t s = my_hist_view.shape[0]
    # perform empirical density conversion
    _cython_empirical_density_hist_int_parallel(s, my_hist_view, empirical_probability_conversion_factor, bin_width)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef _cython_variable_bw_empirical_density_hist(Py_ssize_t view_size, cnp.float64_t[::1] my_hist_view, double empirical_probability_conversion_factor, cnp.float64_t[::1] bin_edge_view):
    # ensure cython compiler knows that the counter in the prange loop is a C integer
    cdef int i
    # loop to perform empirical density conversion
    for i in range(view_size):
        my_hist_view[i] /= (empirical_probability_conversion_factor * (bin_edge_view[i + 1] - bin_edge_view[i]))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef _cython_variable_bw_empirical_density_hist_int(Py_ssize_t view_size, cnp.float64_t[::1] my_hist_view, int empirical_probability_conversion_factor, cnp.float64_t[::1] bin_edge_view):
    # ensure cython compiler knows that the counter in the prange loop is a C integer
    cdef int i
    # loop to perform empirical density conversion
    for i in range(view_size):
        my_hist_view[i] /= (<double>empirical_probability_conversion_factor * (bin_edge_view[i + 1] - bin_edge_view[i]))


def cython_variable_bw_empirical_density_hist(cnp.float64_t[::1] my_hist_view, double empirical_probability_conversion_factor, cnp.float64_t[::1] bin_edge_view):
    # obtain the size
    cdef Py_ssize_t s = my_hist_view.shape[0]
    # perform empirical density conversion
    _cython_variable_bw_empirical_density_hist(s, my_hist_view, empirical_probability_conversion_factor, bin_edge_view)


def cython_variable_bw_empirical_density_hist_int(cnp.float64_t[::1] my_hist_view, int empirical_probability_conversion_factor, cnp.float64_t[::1] bin_edge_view):
    # obtain the size
    cdef Py_ssize_t s = my_hist_view.shape[0]
    # perform empirical density conversion
    _cython_variable_bw_empirical_density_hist_int(s, my_hist_view, empirical_probability_conversion_factor, bin_edge_view)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef _cython_variable_bw_empirical_density_hist_parallel(Py_ssize_t view_size, cnp.float64_t[::1] my_hist_view, double empirical_probability_conversion_factor, cnp.float64_t[::1] bin_edge_view):
    # ensure cython compiler knows that the counter in the prange loop is a C integer
    cdef int i
    # loop to perform empirical density conversion
    for i in prange(view_size, nogil=True):
        my_hist_view[i] /= (empirical_probability_conversion_factor * (bin_edge_view[i + 1] - bin_edge_view[i]))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef _cython_variable_bw_empirical_density_hist_int_parallel(Py_ssize_t view_size, cnp.float64_t[::1] my_hist_view, int empirical_probability_conversion_factor, cnp.float64_t[::1] bin_edge_view):
    # ensure cython compiler knows that the counter in the prange loop is a C integer
    cdef int i
    # loop to perform empirical density conversion
    for i in prange(view_size, nogil=True):
        my_hist_view[i] /= (<double>empirical_probability_conversion_factor * (bin_edge_view[i + 1] - bin_edge_view[i]))


def cython_variable_bw_empirical_density_hist_parallel(cnp.float64_t[::1] my_hist_view, double empirical_probability_conversion_factor, cnp.float64_t[::1] bin_edge_view):
    # obtain the size
    cdef Py_ssize_t s = my_hist_view.shape[0]
    # perform empirical density conversion
    _cython_variable_bw_empirical_density_hist_parallel(s, my_hist_view, empirical_probability_conversion_factor, bin_edge_view)


def cython_variable_bw_empirical_density_hist_int_parallel(cnp.float64_t[::1] my_hist_view, int empirical_probability_conversion_factor, cnp.float64_t[::1] bin_edge_view):
    # obtain the size
    cdef Py_ssize_t s = my_hist_view.shape[0]
    # perform empirical density conversion
    _cython_variable_bw_empirical_density_hist_int_parallel(s, my_hist_view, empirical_probability_conversion_factor, bin_edge_view)
