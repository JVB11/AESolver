import numpy as np
cimport numpy as cnp
cimport cython
from cython.parallel cimport prange


@cython.boundscheck(False)
@cython.nonecheck(False)
cdef _cython_empirical_probability_conversion_no_loop(Py_ssize_t view_size, double alpha):
    # return the empirical probability conversion factor without a loop
    return (<double>view_size * (1.0 + alpha)) - 1.0


def cython_empirical_probability_conversion_no_loop(Py_ssize_t view_size, double alpha):
    return _cython_empirical_probability_conversion_no_loop(view_size, alpha)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef _cython_empirical_probability_conversion_serial_loop(Py_ssize_t view_size, cnp.float64_t[::1] my_data_view, double alpha):
    # ensure cython compiler knows that the counter in the prange loop is a C integer
    cdef int d
    # create the output empirical probability conversion factor
    cdef cnp.float64_t empirical_probability_conversion_fac = 0.0
    # loop to fill binning array using view
    for d in range(view_size):
        empirical_probability_conversion_fac += my_data_view[d] + alpha
    # return the empirical probability conversion factor
    return empirical_probability_conversion_fac - 1.0


def cython_empirical_probability_conversion_serial_loop(cnp.float64_t[::1] my_data, double alpha):
    # store input size
    cdef Py_ssize_t view_size = my_data.shape[0]
    # obtain and return the empirical probability conversion factor
    return _cython_empirical_probability_conversion_serial_loop(view_size, my_data, alpha)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef _cython_empirical_probability_conversion_parallel_loop(Py_ssize_t view_size, cnp.float64_t[::1] my_data_view, double alpha):
    # ensure cython compiler knows that the counter in the prange loop is a C integer
    cdef int d
    # create the output empirical probability conversion factor
    cdef cnp.float64_t empirical_probability_conversion_fac = 0.0
    # loop to fill binning array using view
    for d in prange(view_size, nogil=True):
        empirical_probability_conversion_fac += my_data_view[d] + alpha
    # return the empirical probability conversion factor
    return empirical_probability_conversion_fac - 1.0


def cython_empirical_probability_conversion_parallel_loop(cnp.float64_t[::1] my_data, double alpha):
    # store input size
    cdef Py_ssize_t view_size = my_data.shape[0]
    # obtain and return the empirical probability conversion factor
    return _cython_empirical_probability_conversion_parallel_loop(view_size, my_data, alpha)
