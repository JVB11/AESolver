import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport log
from cython.parallel cimport prange


def create_output_array_view_float(Py_ssize_t view_size):
    # create output array
    cdef cnp.ndarray[cnp.float64_t, ndim=1] out_arr = np.empty(view_size, dtype=np.float64)
    # create output array view for use loop
    cdef cnp.float64_t[::1] arr_view = out_arr
    # return the array and array view
    return out_arr, arr_view


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef _cython_compute_bin_widths(Py_ssize_t bin_width_view_size, cnp.float64_t[::1] my_bin_edge_view, cnp.float64_t[::1] bin_width_view):
    # ensure cython compiler knows that the counter in the prange loop is a C integer
    cdef int i
    # loop to compute the bin widths
    for i in range(bin_width_view_size):
        bin_width_view[i] = my_bin_edge_view[i + 1] - my_bin_edge_view[i]


def cython_compute_bin_widths(cnp.float64_t[::1] my_bin_edge_view):
    # obtain the size for the bin width array
    cdef Py_ssize_t s = my_bin_edge_view.shape[0] - 1
    # generate the output array and corresponding view
    out_arr, arr_view = create_output_array_view_float(s)
    # compute bin widths
    _cython_compute_bin_widths(s, my_bin_edge_view, arr_view)
    # return the array with bin widths
    return out_arr


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef _cython_compute_bin_widths_parallel(Py_ssize_t bin_width_view_size, cnp.float64_t[::1] my_bin_edge_view, cnp.float64_t[::1] bin_width_view):
    # ensure cython compiler knows that the counter in the prange loop is a C integer
    cdef int i
    # loop to compute the bin widths
    for i in prange(bin_width_view_size, nogil=True):
        bin_width_view[i] = my_bin_edge_view[i + 1] - my_bin_edge_view[i]


def cython_compute_bin_widths_parallel(cnp.float64_t[::1] my_bin_edge_view):
    # obtain the size for the bin width array
    cdef Py_ssize_t s = my_bin_edge_view.shape[0] - 1
    # generate the output array and corresponding view
    out_arr, arr_view = create_output_array_view_float(s)
    # compute bin widths
    _cython_compute_bin_widths_parallel(s, my_bin_edge_view, arr_view)
    # return the array with bin widths
    return out_arr
