import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport floor
from cython.parallel cimport prange


# def create_output_array_view_float(Py_ssize_t view_size):
#     # create output array
#     cdef cnp.ndarray[cnp.float64_t, ndim=1] out_arr = np.empty(view_size, dtype=np.float64)
#     # create output array view for use loop
#     cdef cnp.float64_t[::1] arr_view = out_arr
#     # return the array and array view
#     return out_arr, arr_view


def create_output_array_view(Py_ssize_t view_size):
    # create output array
    cdef cnp.ndarray[cnp.int32_t, ndim=1] out_arr = np.empty(view_size, dtype=np.int32)
    # create output array view for use loop
    cdef cnp.int32_t[::1] arr_view = out_arr
    # return the array and array view
    return out_arr, arr_view


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef _cython_binning_serial_loop(Py_ssize_t view_size, cnp.float64_t[::1] my_data_view, cnp.int32_t[::1] binning_view, double lowest_val, double bin_width, double bin_phase):
    # ensure cython compiler knows that the counter in the prange loop is a C integer
    cdef int d
    # loop to fill binning array using view
    for d in range(view_size):
        binning_view[d] = <cnp.int32_t>floor(((my_data_view[d] - lowest_val) / bin_width) - bin_phase)


# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.cdivision(True)
# @cython.nonecheck(False)
# cdef _cython_binning_serial_loop_float(Py_ssize_t view_size, cnp.float64_t[::1] my_data_view, cnp.float64_t[::1] binning_view, double lowest_val, double bin_width, double bin_phase):
#     # ensure cython compiler knows that the counter in the prange loop is a C integer
#     cdef int d
#     # loop to fill binning array using view
#     for d in range(view_size):
#         binning_view[d] = floor(((my_data_view[d] - lowest_val) / bin_width) - bin_phase)


def cython_binning_serial_loop(cnp.float64_t[::1] my_data, double my_data_min, double bin_width, double bin_phase=0.0):
    """Interface function used to call cython binning function that makes use of numpy array views.
    """
    # store the input size
    cdef Py_ssize_t view_size = my_data.shape[0]
    # create output array and view
    out_arr, bin_view = create_output_array_view(view_size)
    # get binning array
    _cython_binning_serial_loop(view_size, my_data, bin_view, my_data_min, bin_width, bin_phase)
    # return new array
    return out_arr


# def cython_binning_serial_loop_float(cnp.float64_t[::1] my_data, double my_data_min, double bin_width, double bin_phase=0.0):
#     """Interface function used to call cython binning function that makes use of numpy array views.
#     """
#     # store the input size
#     cdef Py_ssize_t view_size = my_data.shape[0]
#     # create output array and view
#     out_arr, bin_view = create_output_array_view_float(view_size)
#     # get binning array
#     _cython_binning_serial_loop_float(view_size, my_data, bin_view, my_data_min, bin_width, bin_phase)
#     # return new array
#     return out_arr


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef _cython_binning_parallel_loop(Py_ssize_t view_size, cnp.float64_t[::1] my_data_view, cnp.int32_t[::1] binning_view, double lowest_val, double bin_width, double bin_phase):
    # ensure cython compiler knows that the counter in the prange loop is a C integer
    cdef int i
    # attempt to fill the binning array
    for i in prange(view_size, nogil=True):
        binning_view[i] = <cnp.int32_t>floor(((my_data_view[i] - lowest_val) / bin_width) - bin_phase)


# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.cdivision(True)
# @cython.nonecheck(False)
# cdef _cython_binning_parallel_loop_float(Py_ssize_t view_size, cnp.float64_t[::1] my_data_view, cnp.float64_t[::1] binning_view, double lowest_val, double bin_width, double bin_phase):
#     # ensure cython compiler knows that the counter in the prange loop is a C integer
#     cdef int i
#     # attempt to fill the binning array
#     for i in prange(view_size, nogil=True):
#         binning_view[i] = floor(((my_data_view[i] - lowest_val) / bin_width) - bin_phase)


def cython_binning_parallel_loop(cnp.float64_t[::1] my_data, double my_data_min, double bin_width, double bin_phase=0.0):
    # store the input size
    cdef Py_ssize_t view_size = my_data.shape[0]
    # create output array and view
    out_arr, arr_view = create_output_array_view(view_size)
    # perform binning
    _cython_binning_parallel_loop(view_size, my_data, arr_view, my_data_min, bin_width, bin_phase)
    # return output array
    return out_arr


# def cython_binning_parallel_loop_float(cnp.float64_t[::1] my_data, double my_data_min, double bin_width, double bin_phase=0.0):
#     # store the input size
#     cdef Py_ssize_t view_size = my_data.shape[0]
#     # create output array and view
#     out_arr, arr_view = create_output_array_view_float(view_size)
#     # perform binning
#     _cython_binning_parallel_loop_float(view_size, my_data, arr_view, my_data_min, bin_width, bin_phase)
#     # return output array
#     return out_arr
