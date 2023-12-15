import numpy as np
cimport numpy as cnp
cimport cython
from cython.parallel cimport prange


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef _cython_empirical_probability_hist(Py_ssize_t view_size, cnp.float64_t[::1] my_hist_view, double empirical_probability_conversion_factor):
    # ensure cython compiler knows that the counter in the prange loop is a C integer
    cdef int i
    # loop to perform empirical probability conversion
    for i in range(view_size):
        my_hist_view[i] /= empirical_probability_conversion_factor


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef _cython_empirical_probability_hist_int(Py_ssize_t view_size, cnp.float64_t[::1] my_hist_view, int empirical_probability_conversion_factor):
    # ensure cython compiler knows that the counter in the prange loop is a C integer
    cdef int i
    # loop to perform empirical probability conversion
    for i in range(view_size):
        my_hist_view[i] /= <double>empirical_probability_conversion_factor


def cython_empirical_probability_hist(cnp.float64_t[::1] my_hist_view, double empirical_probability_conversion_factor):
    # obtain the size
    cdef Py_ssize_t s = my_hist_view.shape[0]
    # perform empirical probability conversion
    _cython_empirical_probability_hist(s, my_hist_view, empirical_probability_conversion_factor)


def cython_empirical_probability_hist_int(cnp.float64_t[::1] my_hist_view, int empirical_probability_conversion_factor):
    # obtain the size
    cdef Py_ssize_t s = my_hist_view.shape[0]
    # perform empirical probability conversion
    _cython_empirical_probability_hist_int(s, my_hist_view, empirical_probability_conversion_factor)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef _cython_empirical_probability_hist_parallel(Py_ssize_t view_size, cnp.float64_t[::1] my_hist_view, double empirical_probability_conversion_factor):
    # ensure cython compiler knows that the counter in the prange loop is a C integer
    cdef int i
    # loop to perform empirical probability conversion
    for i in prange(view_size, nogil=True):
        my_hist_view[i] /= empirical_probability_conversion_factor


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef _cython_empirical_probability_hist_int_parallel(Py_ssize_t view_size, cnp.float64_t[::1] my_hist_view, int empirical_probability_conversion_factor):
    # ensure cython compiler knows that the counter in the prange loop is a C integer
    cdef int i
    # loop to perform empirical probability conversion
    for i in prange(view_size, nogil=True):
        my_hist_view[i] /= <double>empirical_probability_conversion_factor


def cython_empirical_probability_hist_parallel(cnp.float64_t[::1] my_hist_view, double empirical_probability_conversion_factor):
    # obtain the size
    cdef Py_ssize_t s = my_hist_view.shape[0]
    # perform empirical probability conversion
    _cython_empirical_probability_hist_parallel(s, my_hist_view, empirical_probability_conversion_factor)


def cython_empirical_probability_hist_int_parallel(cnp.float64_t[::1] my_hist_view, int empirical_probability_conversion_factor):
    # obtain the size
    cdef Py_ssize_t s = my_hist_view.shape[0]
    # perform empirical probability conversion
    _cython_empirical_probability_hist_int_parallel(s, my_hist_view, empirical_probability_conversion_factor)
