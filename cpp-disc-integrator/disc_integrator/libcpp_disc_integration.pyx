# distutils: language=c++

from disc_integrator.libs.libcpp_disc_integration cimport DiscIntegration
from libcpp cimport bool as cpp_bool
from libcpp.string cimport string as cpp_string
cimport cython
cimport numpy as np
import numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
cdef (double, double) _integrals(
    int origi, double[::1] mus, double[::1] angular_function,
    double[::1] first_mu_der, double[::1] second_mu_der,
    cpp_string ld_function_string, cpp_string integration_string,
    int num_threads, cpp_bool paral
    ):
    # create pointer to new DiscIntegration object
    di_ptr = new DiscIntegration(
        origi, &mus[0], &angular_function[0], &first_mu_der[0],
        &second_mu_der[0], ld_function_string,
        integration_string, num_threads, paral
        )
    # use that object to compute the disc integral factors
    cdef double first = di_ptr.compute_first_disc_integral()
    cdef double second = di_ptr.compute_second_disc_integral()
    # delete heap allocated pointer objects
    del di_ptr
    # return the disc integral factors
    return first, second


def get_integrals(
    double[::1] mus, double[::1] angular_function,
    double[::1] first_mu_der, double[::1] second_mu_der,
    str ld_function_string, str integration_string,
    int num_threads, cpp_bool paral
    ):
    # get the array size
    cdef int origi = mus.shape[0]
    # compute the disc factor integrals
    return _integrals(
        origi, mus, angular_function, first_mu_der,
        second_mu_der, ld_function_string.encode('utf-8'),
        integration_string.encode('utf-8'), num_threads, paral
        )
